"""Shared-backbone proxy-task training entry point."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from shared_backbone.proxy_shared_backbone_lite import (
    DEFAULT_EFFICIENTNET_LITE_BACKBONES,
    SharedBackboneLiteProxyNet,
)
from shared_backbone.proxy_audio_training_utils import (
    build_proxy_loader,
    discover_machine_dirs,
    make_feature_domain_mixture,
    make_local_frame_targets,
    nt_xent_loss,
    sanitize_name,
)

try:
    from torchinfo import summary
except Exception:  # torchinfo is optional
    summary = None


MODES = ["ae", "sep_direct", "ce", "arcface", "simclr", "simsiam"]


# -----------------------------------------------------------------------------
# Loss heads
# -----------------------------------------------------------------------------


class ArcMarginProduct(nn.Module):
    """Self-contained ArcFace-style additive angular margin classifier.

    Forward returns scaled margin logits when labels are provided.
    For evaluation without labels, it returns scaled cosine logits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = Fnn.linear(Fnn.normalize(features), Fnn.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


# -----------------------------------------------------------------------------
# Contrastive losses
# -----------------------------------------------------------------------------


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()
    p = Fnn.normalize(p, dim=-1)
    z = Fnn.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def simsiam_loss(p0: torch.Tensor, z1: torch.Tensor, p1: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    return 0.5 * (negative_cosine_similarity(p0, z1) + negative_cosine_similarity(p1, z0))


# -----------------------------------------------------------------------------
# Args / runtime helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Task grid
    parser.add_argument("--mode", type=str, default="ae", choices=MODES + ["sep", "all"])
    parser.add_argument("--target_class", type=str, default="all")
    parser.add_argument("--target_classes", nargs="+", type=str, default=None)
    parser.add_argument("--backbones", nargs="+", type=str, default=None)
    parser.add_argument("--lite_indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    # Data
    parser.add_argument("--target_dir", type=str, default="./asd_dataset_logmel")
    parser.add_argument("--include_aug", action="store_true")
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--segment_frames", type=int, default=160)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument(
        "--matrix_log_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "already_log", "log", "db", "none"],
    )
    parser.add_argument("--feature_scale", type=str, default="db", choices=["db", "ln", "linear"])
    parser.add_argument("--random_crop", action="store_true", default=True)
    parser.add_argument("--no_random_crop", dest="random_crop", action="store_false")

    # Optimization
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--linear_epochs", type=int, default=None,
                        help="ArcFace linear CE fine-tuning epochs. None reuses --epochs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_cpu", type=int, default=32)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--linear_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")

    # Model
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--feature_index", type=int, default=3)
    parser.add_argument("--token_time_mode", type=str, default="upsample", choices=["native", "upsample"])
    parser.add_argument("--token_hidden_dim", type=int, default=128)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--simsiam_pred_hidden_dim", type=int, default=256)
    parser.add_argument("--normalize_projection", action="store_true")

    # Proxy-specific
    parser.add_argument("--snr_db", type=float, default=0.0)
    parser.add_argument("--label_mode", type=str, default="machine", choices=["machine", "domain"])
    parser.add_argument(
        "--classification_scope",
        type=str,
        default="global",
        choices=["global", "per_target"],
        help="For machine-label CE/ArcFace, global is recommended; per_target can become single-class.",
    )
    parser.add_argument(
        "--contrastive_scope",
        type=str,
        default="per_target",
        choices=["per_target", "global"],
        help="Training scope for SimCLR/SimSiam. per_target matches the target_class-based experiments.",
    )
    parser.add_argument(
        "--contrastive_pair_policy",
        type=str,
        default="aug_aug",
        choices=["aug_aug", "original_aug"],
        help="View construction for SimCLR/SimSiam. aug_aug uses two stored augmentations; original_aug uses original+stored augmentation.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--arcface_scale", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margins", nargs="+", type=float, default=None, help="ArcFace margin grid, e.g. --margins 0.6 0.5 0.4 0.3")
    parser.add_argument("--aug_noise_std", type=float, default=0.01, help="Deprecated: stored aug files are used for SimCLR/SimSiam.")
    parser.add_argument("--time_mask_width", type=int, default=12, help="Deprecated: stored aug files are used for SimCLR/SimSiam.")
    parser.add_argument("--freq_mask_width", type=int, default=8, help="Deprecated: stored aug files are used for SimCLR/SimSiam.")

    # Saving / runtime
    parser.add_argument("--save_model_dir", type=str, default="./saved_proxy_lite")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--prune_previous_best", action="store_true",
                        help="delete the previous best checkpoint when a new best is saved")
    parser.add_argument("--checkpoint_format", type=str, default="state_dict", choices=["state_dict", "full"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--print_summary", action="store_true")
    parser.add_argument("--test_name", type=str, default="EfficientNetLite_SharedBackbone")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def resolve_backbones(args: argparse.Namespace) -> List[str]:
    if args.backbones:
        return args.backbones
    out: List[str] = []
    for idx in args.lite_indices:
        if idx < 0 or idx >= len(DEFAULT_EFFICIENTNET_LITE_BACKBONES):
            raise ValueError(f"Invalid EfficientNet-Lite index {idx}. Valid range: 0~4")
        out.append(DEFAULT_EFFICIENTNET_LITE_BACKBONES[idx])
    return out


def resolve_tasks(mode: str) -> List[str]:
    mode = mode.lower()
    if mode == "all":
        return ["ae", "sep_direct", "ce", "arcface", "simclr", "simsiam"]
    if mode == "sep":
        return ["sep_direct"]
    return [mode]


def resolve_target_classes(args: argparse.Namespace) -> List[str]:
    machine_dirs = discover_machine_dirs(args.target_dir)
    all_classes = [p.name for p in machine_dirs]
    if args.target_classes is not None:
        if args.target_classes == ["all"]:
            return all_classes
        return args.target_classes
    if str(args.target_class).lower() == "all":
        return all_classes
    return [args.target_class]


def resolve_margins(args: argparse.Namespace) -> List[float]:
    if args.margins is not None:
        return [float(m) for m in args.margins]
    return [float(args.margin)]


def get_loader_task(task: str) -> str:
    if task == "sep_direct":
        return "sep_direct"
    if task in {"ce", "arcface"}:
        return "clf"
    return task


def get_num_classes(label_mode: str, class_to_idx: Dict[str, int]) -> int:
    if label_mode == "machine":
        return len(class_to_idx)
    if label_mode == "domain":
        return 2
    raise ValueError(label_mode)


def label_key_for(args: argparse.Namespace) -> str:
    return "machine_label" if args.label_mode == "machine" else "domain_label"


def job_targets_for_task(task: str, args: argparse.Namespace, target_classes: List[str]) -> List[str]:
    if task in {"ce", "arcface"}:
        if args.label_mode == "machine" and args.classification_scope == "global":
            return ["__all__"]
        return target_classes
    if task in {"simclr", "simsiam"} and args.contrastive_scope == "global":
        return ["__all__"]
    return target_classes


def set_only_linear_classifier_trainable(model: SharedBackboneLiteProxyNet) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("clf_head.")


def set_all_trainable(model: SharedBackboneLiteProxyNet) -> None:
    for param in model.parameters():
        param.requires_grad = True


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class LiteProxyTrainer:
    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def _safe_name(value: object) -> str:
        return sanitize_name(str(value).strip().strip("_"))

    def _checkpoint_prefix(
        self,
        *,
        args: argparse.Namespace,
        task: str,
        backbone_name: str,
        target_class: str,
        phase: Optional[str] = None,
        margin: Optional[float] = None,
    ) -> str:
        parts = [
            self._safe_name(args.test_name),
            self._safe_name(task),
            self._safe_name(backbone_name),
            f"target{self._safe_name(target_class)}",
        ]
        if task == "arcface" and margin is not None:
            parts.append(f"m{margin:g}")
        parts.extend([
            f"fi{args.feature_index}",
            f"seg{args.segment_frames}",
            f"fs{args.frame_stack}",
            f"batch{args.batch_size}",
        ])
        if phase is not None and phase != task:
            parts.append(self._safe_name(phase))
        return "_".join(parts)

    def _checkpoint_path(
        self,
        *,
        save_dir: Path,
        args: argparse.Namespace,
        task: str,
        backbone_name: str,
        target_class: str,
        epoch_num: int,
        loss_value: float,
        kind: str,
        phase: Optional[str] = None,
        margin: Optional[float] = None,
    ) -> Path:
        prefix = self._checkpoint_prefix(
            args=args,
            task=task,
            backbone_name=backbone_name,
            target_class=target_class,
            phase=phase,
            margin=margin,
        )
        if kind == "best":
            filename = f"{prefix}_best_epoch{epoch_num:03d}_loss{loss_value:.8f}.pth"
        else:
            filename = f"{prefix}_epoch{epoch_num:03d}_loss{loss_value:.8f}.pth"
        return save_dir / filename

    def _save_checkpoint(
        self,
        *,
        path: Path,
        model: SharedBackboneLiteProxyNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        args: argparse.Namespace,
        epoch: int,
        best_loss: float,
        task: str,
        backbone_name: str,
        target_class: str,
        class_to_idx: Dict[str, int],
        phase: Optional[str] = None,
        margin: Optional[float] = None,
        extra_state: Optional[Dict[str, object]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.checkpoint_format == "state_dict" and extra_state is None:
            torch.save(model.state_dict(), path)
            return

        payload: Dict[str, object] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "task": task,
            "phase": phase or task,
            "backbone_name": backbone_name,
            "target_class": target_class,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "feat_dim": model.feat_dim,
            "margin": margin,
        }
        if args.checkpoint_format == "full":
            payload.update({
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            })
        if extra_state:
            payload.update(extra_state)
        torch.save(payload, path)

    def _run_checkpoint_logic(
        self,
        *,
        model: SharedBackboneLiteProxyNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        args: argparse.Namespace,
        epoch_num: int,
        epoch_loss: float,
        best_loss: float,
        best_path: Optional[Path],
        best_weights: Optional[Dict[str, torch.Tensor]],
        save_dir: Path,
        task: str,
        backbone_name: str,
        target_class: str,
        class_to_idx: Dict[str, int],
        phase: str,
        margin: Optional[float] = None,
        extra_state: Optional[Dict[str, object]] = None,
    ) -> Tuple[float, Optional[Path], Optional[Dict[str, torch.Tensor]], bool]:
        is_best = epoch_loss < best_loss
        should_save_interval = args.save_interval > 0 and epoch_num % args.save_interval == 0

        if is_best:
            best_loss = epoch_loss
            if args.prune_previous_best and best_path is not None and best_path.exists():
                try:
                    best_path.unlink()
                except FileNotFoundError:
                    pass
            best_path = self._checkpoint_path(
                save_dir=save_dir,
                args=args,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                epoch_num=epoch_num,
                loss_value=best_loss,
                kind="best",
                phase=phase,
                margin=margin,
            )
            self._save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch_num,
                best_loss=best_loss,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                class_to_idx=class_to_idx,
                phase=phase,
                margin=margin,
                extra_state=extra_state,
            )
            best_weights = copy.deepcopy(model.state_dict())
            logging.info(f"Saved best model: {best_path}")
            with open(save_dir / f"best_model_{phase}.txt", "w", encoding="utf-8") as f:
                f.write(str(best_path) + "\n")

        if should_save_interval:
            epoch_path = self._checkpoint_path(
                save_dir=save_dir,
                args=args,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                epoch_num=epoch_num,
                loss_value=epoch_loss,
                kind="epoch",
                phase=phase,
                margin=margin,
            )
            self._save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch_num,
                best_loss=best_loss,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                class_to_idx=class_to_idx,
                phase=phase,
                margin=margin,
                extra_state=extra_state,
            )
            logging.info(f"Saved interval model: {epoch_path}")

        return best_loss, best_path, best_weights, is_best

    def train_one_epoch(
        self,
        *,
        model: SharedBackboneLiteProxyNet,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        task: str,
        arcface_head: Optional[ArcMarginProduct] = None,
    ) -> Tuple[float, Dict[str, float]]:
        model.train()
        if arcface_head is not None:
            arcface_head.train()

        if len(loader) == 0:
            raise RuntimeError(
                "Shared-backbone DataLoader is empty. "
                f"dataset_size={len(loader.dataset)}, batch_size={loader.batch_size}, drop_last={loader.drop_last}"
            )

        epoch_loss = 0.0
        batch_count = 0
        last_metrics: Dict[str, float] = {}
        ce_loss = nn.CrossEntropyLoss()
        label_key = label_key_for(args)

        for batch_count, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(self.device, args.amp):
                if task == "ae":
                    x = batch["logmel"].float().to(self.device, non_blocking=True)
                    pred = model(x, task="ae")["ae_pred"]
                    target = make_local_frame_targets(x, token_steps=pred.shape[1], frame_stack=model.frame_stack)
                    loss = Fnn.mse_loss(pred, target)
                    last_metrics = {"loss_ae": float(loss.detach().cpu())}

                elif task == "sep_direct":
                    x_t = batch["target_logmel"].float().to(self.device, non_blocking=True)
                    x_nt = batch["nontarget_logmel"].float().to(self.device, non_blocking=True)
                    x_mix, x_nt_scaled, _ = make_feature_domain_mixture(
                        x_t,
                        x_nt,
                        snr_db=args.snr_db,
                        feature_scale=args.feature_scale,
                    )
                    pred = model(x_mix, task="sep_direct")["sep_pred"]
                    target = make_local_frame_targets(x_nt_scaled, token_steps=pred.shape[1], frame_stack=model.frame_stack)
                    loss = Fnn.mse_loss(pred, target)
                    last_metrics = {"loss_sep_direct": float(loss.detach().cpu())}

                elif task == "ce":
                    x = batch["logmel"].float().to(self.device, non_blocking=True)
                    y = batch[label_key].long().to(self.device, non_blocking=True)
                    out = model(x, task="ce")
                    logits = out["logits"]
                    loss = ce_loss(logits, y)
                    acc = (logits.argmax(dim=1) == y).float().mean()
                    last_metrics = {
                        "loss_ce": float(loss.detach().cpu()),
                        "acc": float(acc.detach().cpu()),
                    }

                elif task == "arcface":
                    if arcface_head is None:
                        raise RuntimeError("arcface_head is required for ArcFace training.")
                    x = batch["logmel"].float().to(self.device, non_blocking=True)
                    y = batch[label_key].long().to(self.device, non_blocking=True)
                    feat = model(x, task="encode")["z"]
                    logits = arcface_head(feat, y)
                    loss = ce_loss(logits, y)
                    acc = (logits.argmax(dim=1) == y).float().mean()
                    last_metrics = {
                        "loss_arcface": float(loss.detach().cpu()),
                        "acc": float(acc.detach().cpu()),
                    }

                elif task == "simclr":
                    # Stored augmentation pair from <machine>/aug/*.npy.
                    # The loader returns two positive views matched by original filename.
                    x0 = batch["view0"].float().to(self.device, non_blocking=True)
                    x1 = batch["view1"].float().to(self.device, non_blocking=True)
                    z0 = model(x0, task="simclr")["projection"]
                    z1 = model(x1, task="simclr")["projection"]
                    loss = nt_xent_loss(z0, z1, temperature=args.temperature)
                    last_metrics = {"loss_simclr": float(loss.detach().cpu())}

                elif task == "simsiam":
                    # Stored augmentation pair from <machine>/aug/*.npy.
                    # No online noise/time-mask/freq-mask augmentation is applied here.
                    x0 = batch["view0"].float().to(self.device, non_blocking=True)
                    x1 = batch["view1"].float().to(self.device, non_blocking=True)
                    out0 = model(x0, task="simsiam_predict")
                    out1 = model(x1, task="simsiam_predict")
                    p0, z0 = out0["prediction"], out0["projection"]
                    p1, z1 = out1["prediction"], out1["projection"]
                    loss = simsiam_loss(p0, z1, p1, z0)
                    last_metrics = {"loss_simsiam": float(loss.detach().cpu())}

                else:
                    raise ValueError(f"Unknown task={task}")

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    args.grad_clip,
                )
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        return epoch_loss / max(batch_count, 1), last_metrics

    def train_phase(
        self,
        *,
        model: SharedBackboneLiteProxyNet,
        loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
        task: str,
        backbone_name: str,
        target_class: str,
        class_to_idx: Dict[str, int],
        phase: str,
        epochs: int,
        lr: float,
        save_dir: Path,
        margin: Optional[float] = None,
        arcface_head: Optional[ArcMarginProduct] = None,
    ) -> None:
        params = list(p for p in model.parameters() if p.requires_grad)
        if arcface_head is not None:
            params += list(arcface_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.5)

        best_loss = float("inf")
        best_weights: Optional[Dict[str, torch.Tensor]] = None
        best_path: Optional[Path] = None
        best_epoch: Optional[int] = None

        for epoch in range(epochs):
            epoch_num = epoch + 1
            st_time = time.time()
            epoch_loss, metrics = self.train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                args=args,
                task="ce" if phase == "linear" else task,
                arcface_head=arcface_head,
            )
            scheduler.step()

            extra_state = None
            if arcface_head is not None:
                extra_state = {
                    "arcface_state_dict": arcface_head.state_dict(),
                    "arcface_scale": args.arcface_scale,
                }

            old_best = best_loss
            best_loss, best_path, best_weights, is_best = self._run_checkpoint_logic(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch_num=epoch_num,
                epoch_loss=epoch_loss,
                best_loss=best_loss,
                best_path=best_path,
                best_weights=best_weights,
                save_dir=save_dir,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                class_to_idx=class_to_idx,
                phase=phase,
                margin=margin,
                extra_state=extra_state,
            )
            if is_best and best_loss < old_best:
                best_epoch = epoch_num

            elapsed = time.time() - st_time
            metric_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            logging.info(
                f"task={task}, phase={phase}, target={target_class}, backbone={backbone_name}, "
                f"epoch={epoch_num}, time={elapsed:.2f}s, loss={epoch_loss:.8f}, "
                f"best={best_loss:.8f}, {metric_str}"
            )

        if best_weights is not None:
            model.load_state_dict(best_weights)
            logging.info(
                f"Finished phase={phase}. task={task}, target={target_class}, backbone={backbone_name}, "
                f"best_epoch={best_epoch}, best_loss={best_loss:.8f}, best_model={best_path}"
            )
        else:
            logging.info(f"Finished phase={phase}, but no best checkpoint was saved.")

    def train_job(
        self,
        *,
        model: SharedBackboneLiteProxyNet,
        loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
        task: str,
        backbone_name: str,
        target_class: str,
        class_to_idx: Dict[str, int],
        margin: Optional[float] = None,
    ) -> None:
        safe_task = sanitize_name(task)
        safe_target = sanitize_name(target_class)
        safe_backbone = sanitize_name(backbone_name)
        save_dir = Path(args.save_model_dir) / safe_task / safe_target / safe_backbone
        if task == "arcface" and margin is not None:
            save_dir = save_dir / f"margin_{margin:g}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if task == "arcface":
            set_all_trainable(model)
            arcface_head = ArcMarginProduct(
                in_features=model.feat_dim,
                out_features=get_num_classes(args.label_mode, class_to_idx),
                scale=args.arcface_scale,
                margin=float(margin if margin is not None else args.margin),
            ).to(self.device)
            self.train_phase(
                model=model,
                loader=loader,
                args=args,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                class_to_idx=class_to_idx,
                phase="arcface",
                epochs=args.epochs,
                lr=args.init_lr,
                save_dir=save_dir,
                margin=float(margin if margin is not None else args.margin),
                arcface_head=arcface_head,
            )

            # ArcFace-trained feature extractor + linear CE fine-tuning.
            set_only_linear_classifier_trainable(model)
            self.train_phase(
                model=model,
                loader=loader,
                args=args,
                task=task,
                backbone_name=backbone_name,
                target_class=target_class,
                class_to_idx=class_to_idx,
                phase="linear",
                epochs=int(args.linear_epochs or args.epochs),
                lr=args.linear_lr,
                save_dir=save_dir,
                margin=float(margin if margin is not None else args.margin),
                arcface_head=None,
            )
            set_all_trainable(model)
            return

        self.train_phase(
            model=model,
            loader=loader,
            args=args,
            task=task,
            backbone_name=backbone_name,
            target_class=target_class,
            class_to_idx=class_to_idx,
            phase=task,
            epochs=args.epochs,
            lr=args.init_lr,
            save_dir=save_dir,
            margin=margin,
            arcface_head=None,
        )


# -----------------------------------------------------------------------------
# Job construction
# -----------------------------------------------------------------------------


def build_model(
    *,
    args: argparse.Namespace,
    backbone_name: str,
    num_classes: int,
    device: torch.device,
) -> SharedBackboneLiteProxyNet:
    model = SharedBackboneLiteProxyNet(
        backbone_name=backbone_name,
        pretrained=args.pretrained,
        in_chans=1,
        n_freq=args.n_mels,
        frame_stack=args.frame_stack,
        num_classes=num_classes,
        feature_index=args.feature_index,
        token_time_mode=args.token_time_mode,
        token_hidden_dim=args.token_hidden_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
        simsiam_pred_hidden_dim=args.simsiam_pred_hidden_dim,
        normalize_projection=args.normalize_projection,
    )
    return model.to(device)


def maybe_print_summary(model: SharedBackboneLiteProxyNet, args: argparse.Namespace) -> None:
    if not args.print_summary:
        return
    if summary is None:
        logging.warning("torchinfo is not installed; skip summary.")
        return
    try:
        summary(model, input_size=(1, 1, args.n_mels, args.segment_frames), depth=3)
    except Exception as e:
        logging.warning(f"torchinfo.summary failed: {e}")


def run_one_job(
    *,
    trainer: LiteProxyTrainer,
    args: argparse.Namespace,
    task: str,
    target_class: str,
    backbone_name: str,
    device: torch.device,
    margin: Optional[float] = None,
) -> None:
    loader_task = get_loader_task(task)
    loader, class_to_idx = build_proxy_loader(
        target_dir=args.target_dir,
        target_class=target_class,
        task=loader_task,
        batch_size=args.batch_size,
        n_cpu=args.n_cpu,
        n_mels=args.n_mels,
        n_frame=args.frame_stack,
        segment_frames=args.segment_frames,
        matrix_log_mode=args.matrix_log_mode,
        include_aug=args.include_aug,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        random_crop=args.random_crop,
        contrastive_pair_policy=args.contrastive_pair_policy,
    )

    num_classes = get_num_classes(args.label_mode, class_to_idx)
    if task in {"ce", "arcface"} and args.label_mode == "machine" and target_class != "__all__":
        logging.warning(
            "machine-label CE/ArcFace is being trained on target_class=%s only. "
            "This can become a single-class classifier. Prefer --classification_scope global.",
            target_class,
        )

    model = build_model(args=args, backbone_name=backbone_name, num_classes=num_classes, device=device)
    maybe_print_summary(model, args)

    logging.info(
        f"Start training: task={task}, target={target_class}, backbone={backbone_name}, "
        f"num_classes={num_classes}, n_files={len(loader.dataset)}, margin={margin}"
    )
    trainer.train_job(
        model=model,
        loader=loader,
        args=args,
        task=task,
        backbone_name=backbone_name,
        target_class=target_class,
        class_to_idx=class_to_idx,
        margin=margin,
    )


def main(args: argparse.Namespace) -> None:
    Path(args.save_model_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tasks = resolve_tasks(args.mode)
    target_classes = resolve_target_classes(args)
    backbones = resolve_backbones(args)
    margins = resolve_margins(args)

    logging.info(args)
    logging.info(f"Resolved tasks: {tasks}")
    logging.info(f"Resolved target classes: {target_classes}")
    logging.info(f"Resolved EfficientNet-Lite backbones: {backbones}")

    trainer = LiteProxyTrainer(device=device)

    for backbone_name in backbones:
        for task in tasks:
            targets = job_targets_for_task(task, args, target_classes)
            if task == "arcface":
                for margin in margins:
                    for target_class in targets:
                        run_one_job(
                            trainer=trainer,
                            args=args,
                            task=task,
                            target_class=target_class,
                            backbone_name=backbone_name,
                            device=device,
                            margin=margin,
                        )
            else:
                for target_class in targets:
                    run_one_job(
                        trainer=trainer,
                        args=args,
                        task=task,
                        target_class=target_class,
                        backbone_name=backbone_name,
                        device=device,
                        margin=None,
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main(parse_args())
