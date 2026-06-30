"""Autoencoder proxy-task training entry point."""

import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn

try:
    from torchinfo import summary
except Exception:  # torchinfo is optional for public reruns
    summary = None

from ae_baseline_utils import ae_dataset, AENet, loss_function_mahala, cov_v

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AE proxy models on file-wise log-mel npy features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--target_class", type=str, default="valve", help="single target class to train")
    parser.add_argument("--target_classes", nargs="+", default=None, help="optional explicit target-class list")
    parser.add_argument("--save_model_dir", type=str, default="./saved_exp/ae", help="dir of saved model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_cpu", type=int, default=32)
    parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--target_dir", type=str, default="./asd_dataset_logmel", help="dir of dataset")
    parser.add_argument(
        "--matrix_log_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "already_log", "log", "db", "none"],
        help="Scale handling for AE npy files. Use auto/raw for raw mel-power and already_log for log/db exports.",
    )
    parser.add_argument("--latent_dims", nargs="+", type=int, default=[4, 8, 16],
                        help="latent dimensions, mapped to AENet comp_feat")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 128, 256],
                        help="hidden dimensions, mapped to AENet lin_feat")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="save epoch checkpoint every N epochs. Set 0 or negative to disable.")
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
    def __init__(self, train_ds, device: torch.device):
        self.train_ds = train_ds
        self.gpu_id = device

    def _update_covariance(self, model, loader):
        """Update covariance buffers before checkpoint saving."""
        model.eval()

        cov_x_source = torch.zeros((128, 128), device=self.gpu_id, dtype=torch.float32)
        cov_x_target = cov_x_source.clone().detach()
        cov_x_all = cov_x_source.clone().detach()
        num_source = 0
        num_target = 0

        with torch.no_grad():
            for data in loader:
                spec = data[0].type(torch.FloatTensor).to(self.gpu_id).view(-1, 128 * 5)
                file_name = data[1]

                is_target_list = ["target" in data_name for data_name in file_name]
                is_source_list = np.logical_not(is_target_list).tolist()
                n_source = is_source_list.count(True)
                n_target = is_target_list.count(True)

                sep_out, _ = model(spec)

                n = int(sep_out.shape[0] / len(is_target_list))
                is_source_list = np.repeat(is_source_list, n)
                is_target_list = np.repeat(is_target_list, n)
                _, cov_diff_source, cov_diff_target, cov_diff_all = loss_function_mahala(
                    recon_x=sep_out,
                    x=spec,
                    block_size=128,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list
                )
                cov_x_source_batch = cov_v(
                    diff=cov_diff_source,
                    num=1
                )
                cov_x_all_batch = cov_v(cov_diff_all, 1)
                cov_x_source += cov_x_source_batch.clone().detach()
                cov_x_all += cov_x_all_batch.clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target_batch = cov_v(
                        diff=cov_diff_target,
                        num=1
                    )
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target

        if num_source > 1:
            cov_x_source /= num_source - 1
        if num_target == 0:
            cov_x_target = cov_x_source.clone().detach()
        elif num_target > 1:
            cov_x_target /= num_target - 1
        if (num_source + num_target) > 0:
            cov_x_all /= (num_source + num_target)

        model.cov_source.data = cov_x_source
        model.cov_target.data = cov_x_target
        model.cov_all.data = cov_x_all
        model.train()

    def train(self, model, loader, args, latent_dim, hidden_dim):
        if len(loader) == 0:
            raise RuntimeError(
                "AE DataLoader is empty. "
                f"dataset_size={len(loader.dataset)}, batch_size={loader.batch_size}, drop_last={loader.drop_last}"
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        loss_l2 = torch.nn.MSELoss()
        ae_best = float("inf")
        best_model_weights = None
        best_epoch = None
        best_save_path = None

        save_name_prefix = (
            f"{args.target_class}_comp{latent_dim}lin{hidden_dim}"
            f"_baseline_batch{args.batch_size}"
        )

        for epoch in range(args.epochs):
            model.train()
            epoch_num = epoch + 1
            st_time = time.time()
            ae_loss = 0.0
            batch_count = 0

            for batch_count, data in enumerate(loader, start=1):
                spec = data[0].type(torch.FloatTensor).to(self.gpu_id)

                # For old AE adopter,
                spec = spec.view(-1, spec.shape[2])

                optimizer.zero_grad()
                sep_out, _ = model(spec)
                ae_l2 = loss_l2(sep_out, spec)

                ae_l2.backward()
                optimizer.step()
                ae_loss += ae_l2.item() / sep_out.shape[0]

            ae_loss = ae_loss / max(batch_count, 1)
            epoch_loss = ae_loss
            scheduler.step()
            optimizer.zero_grad()

            is_best = ae_loss < ae_best
            if is_best:
                ae_best = ae_loss
                best_epoch = epoch_num

            should_save_interval = args.save_interval > 0 and epoch_num % args.save_interval == 0
            if is_best or should_save_interval:
                # Checkpoints use the same covariance update logic as the original best-model save path.
                self._update_covariance(model, loader)

                if is_best:
                    # Keep only the latest best-loss checkpoint for this latent/hidden combination.
                    if args.prune_previous_best and best_save_path is not None and os.path.exists(best_save_path):
                        os.remove(best_save_path)
                    best_save_path = os.path.join(
                        args.save_model_dir,
                        f"{save_name_prefix}_best_epoch{epoch_num:03d}_ae{ae_best:.8f}.pth"
                    )
                    torch.save(model.state_dict(), best_save_path)
                    best_model_weights = copy.deepcopy(model.state_dict())
                    logging.info(f"Saved best model: {best_save_path}")

                if should_save_interval:
                    epoch_save_path = os.path.join(
                        args.save_model_dir,
                        f"{save_name_prefix}_epoch{epoch_num:03d}_ae{epoch_loss:.8f}.pth"
                    )
                    torch.save(model.state_dict(), epoch_save_path)
                    logging.info(f"Saved interval model: {epoch_save_path}")

            logging.info(
                f'{epoch_num} epoch, {(str(time.time() - st_time))[:5]}seconds')
            logging.info(
                f'{epoch_num} epoch, total loss : {(str(epoch_loss))[:8]}, best : {(str(ae_best))[:8]}')

        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
            logging.info(
                f'Unsupervised train finished. Best epoch={best_epoch}, best loss={ae_best:.8f}, '
                f'best model={best_save_path}'
            )
        else:
            logging.info('Unsupervised train finished, but no epoch was trained; no best model was saved.')

    def main_tr(self, args):
        for latent_dim in args.latent_dims:
            for hidden_dim in args.hidden_dims:
                logging.info(
                    f"Start training target={args.target_class}, "
                    f"latent_dim={latent_dim}, hidden_dim={hidden_dim}"
                )

                model = AENet(
                    input_dim=128 * 5,
                    block_size=128,
                    lin_feat=hidden_dim,
                    comp_feat=latent_dim
                )

                maybe_print_summary(model, [(1, 128 * 5)], args)

                self.train(
                    model.to(self.gpu_id),
                    self.train_ds,
                    args,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim
                )


def main(rank: int, world_size: int, args):
    os.makedirs(args.save_model_dir, exist_ok=True)

    logging.info(args)
    device = resolve_device(args.device)

    train_ds = ae_dataset(
        args.target_dir, args.batch_size, args.n_cpu,
        args.target_class,
        matrix_log_mode=args.matrix_log_mode,
    )
    trainer = Trainer(train_ds, device)
    trainer.main_tr(args=args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    configure_runtime(args)
    for target_class in iter_target_classes(args):
        run_args = copy.copy(args)
        run_args.target_class = target_class
        main(rank=0, world_size=0, args=run_args)
