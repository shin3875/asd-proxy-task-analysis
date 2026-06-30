"""Classification and ArcFace proxy-task training entry point."""

import argparse
import copy
import glob
import logging
import os
import time

import torch
import torch.nn

try:
    from torchinfo import summary
except Exception:  # torchinfo is optional for public reruns
    summary = None

import dataloader
from models.generator_comp import ArcFaceClassifier, arcface_loss
from models.resnet_oth import ResNet


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

MODEL_SETTINGS = {
    "resnet18": ("ResNet18", 512),
    "resnet34": ("ResNet34", 512),
    "resnet50": ("ResNet50", 2048),
    "resnet101": ("ResNet101", 2048),
    "resnet152": ("ResNet152", 2048),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CE or ArcFace proxy classifiers on DCASE-style wav data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--target_class", type=str, default="pump", help="single target class to train")
    parser.add_argument("--target_classes", nargs="+", default=None, help="optional explicit target-class list")
    parser.add_argument("--mode", type=str, default="arcface", choices=["ce", "arcface"])
    parser.add_argument("--save_model_dir", type=str, default="./saved_exp/classifier", help="dir of saved model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_cpu", type=int, default=None, help="data-loader workers; defaults to batch_size. Use 0 for main-process loading")
    parser.add_argument("--input_len", type=int, default=160000)
    parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
    parser.add_argument("--test_name", type=str, default=None,
                        help="checkpoint name prefix. Defaults to the selected ResNet display name.")
    parser.add_argument("--resnet", type=str, default="resnet18", choices=sorted(MODEL_SETTINGS))
    parser.add_argument("--resnets", nargs="+", default=None, choices=sorted(MODEL_SETTINGS),
                        help="optional explicit ResNet list")
    parser.add_argument("--init_lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--target_dir", type=str, default="./asd_dataset", help="dir of dataset")
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace margin")
    parser.add_argument("--margins", nargs="+", type=float, default=None, help="optional ArcFace margin list")
    parser.add_argument("--save_interval", type=int, default=150,
                        help="save checkpoint every N epochs. Set 0 or negative to disable interval saves.")
    parser.add_argument("--prune_previous_best", action="store_true",
                        help="delete the previous best checkpoint when a new best is saved")
    parser.add_argument("--cuda_visible_devices", default=None,
                        help="optional CUDA_VISIBLE_DEVICES value for this run")
    parser.add_argument("--device", default="auto",
                        help="torch device, e.g. auto, cuda:0, or cpu")
    parser.add_argument("--print_summary", action="store_true",
                        help="print torchinfo model summary when torchinfo is installed")
    return parser.parse_args()


def iter_target_classes(args: argparse.Namespace) -> list[str]:
    return args.target_classes if args.target_classes is not None else [args.target_class]


def iter_resnets(args: argparse.Namespace) -> list[str]:
    return args.resnets if args.resnets is not None else [args.resnet]


def iter_margins(args: argparse.Namespace) -> list[float]:
    if args.mode == "arcface":
        return args.margins if args.margins is not None else [float(args.margin)]
    return [float(args.margin)]


def configure_runtime(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def maybe_print_summary(model, input_size, args) -> None:
    if not args.print_summary:
        return
    if summary is None:
        logging.warning("torchinfo is not installed; skip summary.")
        return
    try:
        summary(model, input_size)
    except Exception as exc:
        logging.warning(f"torchinfo.summary failed: {exc}")


class Trainer:
    def __init__(self, train_ds, device: torch.device, class_num):
        self.train_ds = train_ds
        self.gpu_id = device
        self.class_num = class_num

    @staticmethod
    def _safe_name(value):
        """Make a string safe enough for checkpoint file names."""
        return str(value).strip().strip('_').replace(os.sep, '-').replace(' ', '')

    def _checkpoint_prefix(self, args, phase=None):
        """
        Build a checkpoint prefix that supports both CE and ArcFace.

        - CE: target_class is intentionally excluded from the file name.
        - ArcFace: target_class is included in the file name.
        """
        mode = args.mode.lower()
        parts = [
            self._safe_name(args.test_name),
            mode,
        ]

        if mode == 'arcface':
            parts += [
                f"target{self._safe_name(args.target_class)}",
                f"m{self._safe_name(args.margin)}",
            ]

        parts.append(f"batch{args.batch_size}")

        if phase is not None and phase != mode:
            parts.append(self._safe_name(phase))

        return "_".join(part for part in parts if part)

    def _checkpoint_path(self, args, epoch_num, loss_value, kind='epoch', phase=None):
        prefix = self._checkpoint_prefix(args, phase=phase)
        loss_str = f"{loss_value:.8f}"

        if kind == 'best':
            filename = f"{prefix}_best_epoch{epoch_num:03d}_loss{loss_str}.pth"
        else:
            filename = f"{prefix}_epoch{epoch_num:03d}_loss{loss_str}.pth"

        return os.path.join(args.save_model_dir, filename)

    def _save_checkpoint(self, model, args, epoch_num, loss_value, kind='epoch', phase=None):
        os.makedirs(args.save_model_dir, exist_ok=True)
        save_path = self._checkpoint_path(
            args=args,
            epoch_num=epoch_num,
            loss_value=loss_value,
            kind=kind,
            phase=phase,
        )
        torch.save(model.state_dict(), save_path)
        return save_path

    def _train_ce_epoch(self, model, loader, ce_loss, optimizer):
        if len(loader) == 0:
            raise RuntimeError("Classifier DataLoader is empty. Check dataset paths, batch size, and drop_last settings.")
        model.train()
        train_loss = 0.0

        for data in loader:
            target = data[0].type(torch.FloatTensor).to(self.gpu_id)
            labels = data[1].to(self.gpu_id)

            optimizer.zero_grad()
            ori_out, _ = model(target)
            loss = ce_loss(ori_out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        return train_loss / len(loader)

    def _train_arcface_epoch(self, model, arcclf, loader, optimizer, args):
        if len(loader) == 0:
            raise RuntimeError("ArcFace DataLoader is empty. Check dataset paths, batch size, and drop_last settings.")
        model.train()
        arcclf.train()
        train_loss = 0.0

        for data in loader:
            target = data[0].type(torch.FloatTensor).to(self.gpu_id)
            labels = data[1].to(self.gpu_id)

            optimizer.zero_grad()
            _, ori_feat = model(target)
            arc_out0 = arcclf(ori_feat)
            loss = arcface_loss(arc_out0, labels, self.class_num, m=args.margin)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        return train_loss / len(loader)

    def train(self, model, loader, args):
        args.mode = args.mode.lower()
        if args.mode not in ['ce', 'arcface']:
            raise ValueError(f"args.mode must be either 'ce' or 'arcface', but got: {args.mode}")

        ce_loss = torch.nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_model_weights = copy.deepcopy(model.state_dict())
        best_checkpoint_path = None

        # ------------------------------------------------------------------
        # Phase 1: CE or ArcFace training
        # ------------------------------------------------------------------
        if args.mode == 'arcface':
            arcclf = ArcFaceClassifier(
                emb_size=args.channel,
                output_classes=self.class_num,
                gpu_id=self.gpu_id,
            ).to(self.gpu_id)
            params_to_train = list(model.parameters()) + list(arcclf.parameters())
            phase = 'arcface'
        else:
            arcclf = None
            params_to_train = list(model.parameters())
            phase = 'ce'

        optimizer = torch.optim.AdamW(params_to_train, lr=args.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay_epoch, gamma=0.5
        )

        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            st_time = time.time()

            if args.mode == 'arcface':
                train_loss = self._train_arcface_epoch(model, arcclf, loader, optimizer, args)
            else:
                train_loss = self._train_ce_epoch(model, loader, ce_loss, optimizer)

            scheduler.step()

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                new_best_path = self._save_checkpoint(
                    model=model,
                    args=args,
                    epoch_num=epoch_num,
                    loss_value=best_loss,
                    kind='best',
                    phase=phase,
                )
                if args.prune_previous_best and best_checkpoint_path is not None and best_checkpoint_path != new_best_path:
                    try:
                        os.remove(best_checkpoint_path)
                    except FileNotFoundError:
                        pass
                best_checkpoint_path = new_best_path

            if args.save_interval > 0 and epoch_num % args.save_interval == 0:
                self._save_checkpoint(
                    model=model,
                    args=args,
                    epoch_num=epoch_num,
                    loss_value=train_loss,
                    kind='epoch',
                    phase=phase,
                )

            logging.info(
                f'{epoch_num} epoch, {(str(time.time() - st_time))[:5]}seconds')
            logging.info(
                f'{epoch_num} epoch, loss : {(str(train_loss))[:8]}, best : {(str(best_loss))[:8]}')

        if args.mode == 'ce':
            logging.info(
                f'Cross-entropy based train finished. Best loss: {best_loss:.8f}, checkpoint: {best_checkpoint_path}')
            model.load_state_dict(best_model_weights)
            return

        # ------------------------------------------------------------------
        # Phase 2: ArcFace-trained feature extractor + linear CE fine-tuning
        # ------------------------------------------------------------------
        model.load_state_dict(best_model_weights)
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer_finetune = torch.optim.AdamW(params_to_update, lr=0.0001)
        scheduler_finetune = torch.optim.lr_scheduler.StepLR(
            optimizer_finetune, step_size=args.decay_epoch, gamma=0.5
        )

        linear_best_loss = float('inf')
        linear_best_model_weights = copy.deepcopy(model.state_dict())
        linear_best_checkpoint_path = None
        linear_phase = 'linear'

        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            st_time = time.time()

            train_loss = self._train_ce_epoch(
                model=model,
                loader=loader,
                ce_loss=ce_loss,
                optimizer=optimizer_finetune,
            )
            scheduler_finetune.step()

            if train_loss < linear_best_loss:
                linear_best_loss = train_loss
                linear_best_model_weights = copy.deepcopy(model.state_dict())
                new_best_path = self._save_checkpoint(
                    model=model,
                    args=args,
                    epoch_num=epoch_num,
                    loss_value=linear_best_loss,
                    kind='best',
                    phase=linear_phase,
                )
                if args.prune_previous_best and linear_best_checkpoint_path is not None and linear_best_checkpoint_path != new_best_path:
                    try:
                        os.remove(linear_best_checkpoint_path)
                    except FileNotFoundError:
                        pass
                linear_best_checkpoint_path = new_best_path

            if args.save_interval > 0 and epoch_num % args.save_interval == 0:
                self._save_checkpoint(
                    model=model,
                    args=args,
                    epoch_num=epoch_num,
                    loss_value=train_loss,
                    kind='epoch',
                    phase=linear_phase,
                )

            logging.info(f'{epoch_num} epoch - Linear training, {(str(time.time() - st_time))[:5]}seconds')
            logging.info(
                f'{epoch_num} epoch, loss : {(str(train_loss))[:8]}, best : {(str(linear_best_loss))[:8]}')

        logging.info(
            f'ArcFace based train finished. Best linear loss: {linear_best_loss:.8f}, checkpoint: {linear_best_checkpoint_path}')
        model.load_state_dict(linear_best_model_weights)

    def main_tr(self, class_num, args):
        args.mode = args.mode.lower()
        if args.mode not in ['ce', 'arcface']:
            raise ValueError(f"args.mode must be either 'ce' or 'arcface', but got: {args.mode}")

        for resnet_type in iter_resnets(args):
            test_name, channel = MODEL_SETTINGS[resnet_type]
            model = ResNet(num_class=class_num, resnet_type=resnet_type)
            maybe_print_summary(model, [(1, 1, 128, 313)], args)
            run_args = copy.copy(args)
            run_args.test_name = args.test_name or test_name
            run_args.channel = channel
            self.train(model.to(self.gpu_id), self.train_ds, run_args)


def main(rank: int, world_size: int, args):
    os.makedirs(args.save_model_dir, exist_ok=True)

    args.mode = args.mode.lower()
    if args.mode not in ['ce', 'arcface']:
        raise ValueError(f"args.mode must be either 'ce' or 'arcface', but got: {args.mode}")

    logging.info(args)
    device = resolve_device(args.device)

    dirs = sorted(glob.glob(os.path.abspath("{base}/*".format(base=args.target_dir))))
    dirs = [f for f in dirs if os.path.isdir(f)]

    class_list = [os.path.basename(f) for f in dirs]

    num_workers = args.batch_size if args.n_cpu is None else int(args.n_cpu)
    train_ds = dataloader.class_data_arcface(
        args.target_dir, args.batch_size, num_workers,
        args.target_class, class_list=class_list, snr_list=[3, 4, 5, 6, 7, 8, 9, 10]
    )
    trainer = Trainer(train_ds, device, len(class_list))
    trainer.main_tr(class_num=len(class_list), args=args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    configure_runtime(args)
    for target_class in iter_target_classes(args):
        for margin in iter_margins(args):
            run_args = copy.copy(args)
            run_args.target_class = target_class
            run_args.margin = margin
            main(rank=0, world_size=0, args=run_args)
