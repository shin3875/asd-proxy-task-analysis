"""Separation proxy-task training entry point."""

import argparse
import copy
import glob
import logging
import os
import re
import time

import torch
import torch.nn

try:
    from torchinfo import summary
except Exception:  # torchinfo is optional for public reruns
    summary = None

import dataloader
from models.generator_comp import TSCNet_Cont

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separation proxy models on DCASE-style wav data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--target_class", type=str, default="valve", help="single target class to train")
    parser.add_argument("--target_classes", nargs="+", default=None, help="optional explicit target-class list")
    parser.add_argument("--save_model_dir", type=str, default="./saved_exp/separation",
                        help="dir of saved model")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--n_cpu", type=int, default=None, help="data-loader workers; defaults to batch_size. Use 0 for main-process loading")
    parser.add_argument("--input_len", type=int, default=32000)
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--channels", nargs="+", type=int, default=None,
                        help="optional channel-width grid, e.g. 64 128")
    parser.add_argument("--cb", type=int, default=4, choices=[0, 1, 2, 4],
                        help="number of conformer blocks")
    parser.add_argument("--cbs", nargs="+", type=int, default=None, choices=[0, 1, 2, 4],
                        help="optional conformer-block grid, e.g. 0 1 2 4")
    parser.add_argument("--decay_epoch", type=int, default=40, help="epoch from which to start lr decay")
    parser.add_argument("--test_name", type=str, default="target_snr_m5to5", help="test name in checkpoint file")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--target_dir", type=str, default="./asd_dataset", help="dir of dataset")
    parser.add_argument("--margin", type=float, default=0.1, help="negative contrastive margin")
    parser.add_argument("--snr_list", nargs="+", type=float, default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        help="SNR list for separation mixture generation")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="save epoch checkpoint every N epochs. Default 1 preserves the original behavior.")
    parser.add_argument("--cuda_visible_devices", default=None,
                        help="optional CUDA_VISIBLE_DEVICES value for this run")
    parser.add_argument("--device", default="auto",
                        help="torch device, e.g. auto, cuda:0, or cpu")
    parser.add_argument("--print_summary", action="store_true",
                        help="print torchinfo model summary when torchinfo is installed")
    return parser.parse_args()


def iter_target_classes(args: argparse.Namespace) -> list[str]:
    return args.target_classes if args.target_classes is not None else [args.target_class]


def iter_channels(args: argparse.Namespace) -> list[int]:
    return args.channels if args.channels is not None else [args.channel]


def iter_cbs(args: argparse.Namespace) -> list[int]:
    return args.cbs if args.cbs is not None else [args.cb]


def configure_runtime(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)


def normalize_test_name_for_cb(test_name: str) -> str:
    """Keep CB metadata in one canonical filename token generated from --cb."""
    clean = re.sub(r"(^|_)\d+cb(?=_|$)", "_", str(test_name), flags=re.IGNORECASE)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "target_snr_m5to5"


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
    def __init__(self, train_ds, device: torch.device):
        self.n_fft = 400
        self.hop = 200
        self.train_ds = train_ds
        self.gpu_id = device
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def feature_maker(self, signal):
        signal_spec = torch.stft(
            signal,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=True
        ).unsqueeze(dim=1)
        signal_spec = torch.cat([signal_spec.real, signal_spec.imag], dim=1)

        return signal_spec

    def mag_from_ri(self, x, eps=1e-8):
        # x: [B, 2, F, T]
        real = x[:, 0]
        imag = x[:, 1]
        return torch.sqrt(real * real + imag * imag + eps)  # [B, F, T]

    def train(self, model, loader, args):
        if len(loader) == 0:
            raise RuntimeError("Separation DataLoader is empty. Check target/non-target files and batch size.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        loss_l1 = torch.nn.L1Loss()
        loss_l2 = torch.nn.MSELoss()
        sep_best = float("inf")
        best_model_weights = None
        best_name = None
        model.train()
        for epoch in range(args.epochs):

            st_time = time.time()
            save_name = (
                f"{args.target_class}_{args.test_name}_batch{args.batch_size}_"
                f"{args.cb}cb_{args.channel}ch_epoch{epoch}"
            )

            sep_loss = 0.0
            contrastive_loss = 0.0

            cont_train = False
            sep_train = True
            for data in loader:
                if sep_train:
                    mixture = data[4].type(torch.FloatTensor).to(self.gpu_id)
                    target = data[0].type(torch.FloatTensor).to(self.gpu_id)

                    mixture_spec = self.feature_maker(mixture)
                    target_spec = self.feature_maker(target)

                    optimizer.zero_grad()
                    sep_out, _ = model(mixture_spec)

                    mag_pred = self.mag_from_ri(sep_out)
                    mag_true = self.mag_from_ri(target_spec)

                    loss_mag = loss_l1(torch.log(mag_pred + 1e-6), torch.log(mag_true + 1e-6)) \
                               + loss_l2(mag_pred.pow(0.3), mag_true.pow(0.3))

                    loss_cplx = torch.nn.SmoothL1Loss(beta=0.5)(sep_out, target_spec)

                    sep_sum = loss_mag + 0.05 * loss_cplx

                    sep_sum.backward()
                    optimizer.step()
                    sep_loss += sep_sum.item()

            sep_loss = sep_loss / len(loader)
            epoch_loss = contrastive_loss + sep_loss
            scheduler.step()

            is_best = sep_loss < sep_best and sep_train
            if is_best:
                sep_best = sep_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_name = f'{save_name}_loss{str(epoch_loss)[:5]}_sep{str(sep_loss)[:5]}_best.pth'

            if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                epoch_name = f'{save_name}_loss{str(epoch_loss)[:5]}_sep{str(sep_loss)[:5]}.pth'
                torch.save(model.state_dict(), os.path.join(args.save_model_dir, epoch_name))

            logging.info(
                f'{epoch} epoch, {(str(time.time() - st_time))[:5]}seconds')
            logging.info(
                f'{epoch} epoch, total loss : {(str(epoch_loss))[:5]}, best : {(str(sep_best))[:5]}')
            if sep_train:
                logging.info(
                    f'{epoch} epoch, separation loss : {(str(sep_loss))[:5]}, best : {(str(sep_best))[:5]}')
            if cont_train:
                logging.info(
                    f'{epoch} epoch, cont loss : {(str(contrastive_loss))[:5]}')
        if best_model_weights is not None and best_name is not None:
            model.load_state_dict(best_model_weights)
            torch.save(model.state_dict(), os.path.join(args.save_model_dir, best_name))
            logging.info(f"Saved best model: {os.path.join(args.save_model_dir, best_name)}")
        else:
            logging.info("No separation checkpoint was saved because no epoch was trained.")

    def main_tr(self, args):
        model = TSCNet_Cont(num_channel=args.channel, num_features=self.n_fft // 2 + 1, cb=args.cb)

        maybe_print_summary(model, [(1, 2, 201, 481)], args)

        self.train(model.to(self.gpu_id), self.train_ds, args)


def main(rank: int, world_size: int, args):
    os.makedirs(args.save_model_dir, exist_ok=True)

    logging.info(args)
    device = resolve_device(args.device)

    dirs = sorted(glob.glob(os.path.abspath("{base}/*".format(base=args.target_dir))))
    dirs = [f for f in dirs if os.path.isdir(f)]
    class_list = [os.path.basename(f) for f in dirs]
    num_workers = args.batch_size if args.n_cpu is None else int(args.n_cpu)
    train_ds = dataloader.class_data_sep(
        args.target_dir,
        args.batch_size,
        num_workers,
        args.target_class,
        class_list=class_list,
        snr_list=args.snr_list,
        input_len=args.input_len,
    )
    trainer = Trainer(train_ds, device)
    trainer.main_tr(args=args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    configure_runtime(args)
    for target_class in iter_target_classes(args):
        for channel in iter_channels(args):
            for cb in iter_cbs(args):
                run_args = copy.copy(args)
                run_args.target_class = target_class
                run_args.channel = channel
                run_args.cb = cb
                run_args.test_name = normalize_test_name_for_cb(run_args.test_name)
                main(rank=0, world_size=0, args=run_args)
