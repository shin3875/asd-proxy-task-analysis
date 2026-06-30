# proxy_shared_backbone_lite.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("This code requires timm. Install with: pip install timm") from e


DEFAULT_EFFICIENTNET_LITE_BACKBONES = [
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
]


class TokenMLP(nn.Module):
    """Token-wise MLP head.

    Input:
        tokens: [B, T_token, C]
    Output:
        y: [B, T_token, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        final_activation: Optional[str] = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.final_activation is None:
            return y
        if self.final_activation == "sigmoid":
            return torch.sigmoid(y)
        raise ValueError(f"Unknown final_activation: {self.final_activation}")


class ProjectionHead(nn.Module):
    """Small projection head for SimCLR / SimSiam.

    By default, the head returns raw vectors. Loss functions normalize internally.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.normalize_output = bool(normalize_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        if self.normalize_output:
            y = F.normalize(y, dim=-1)
        return y


class SimSiamPredictor(nn.Module):
    """Small predictor head used only for SimSiam."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, out_dim: Optional[int] = None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _candidate_backbone_names(backbone_name: str) -> list[str]:
    """Return timm model-name fallbacks for EfficientNet-Lite variants."""
    names = [backbone_name]
    if backbone_name.endswith(".in1k"):
        names.append(backbone_name.replace(".in1k", ""))
    elif backbone_name.startswith("tf_efficientnet_lite") and "." not in backbone_name:
        names.append(f"{backbone_name}.in1k")
    # remove duplicates while preserving order
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def create_timm_feature_backbone(
    backbone_name: str,
    *,
    pretrained: bool,
    in_chans: int,
    feature_index: int,
):
    """Create a timm features_only backbone with fallback model names.

    Some timm versions expose EfficientNet-Lite as `tf_efficientnet_lite0`,
    while newer model registries may use `tf_efficientnet_lite0.in1k`.
    """
    last_error: Optional[Exception] = None
    for name in _candidate_backbone_names(backbone_name):
        try:
            model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_chans,
                out_indices=(feature_index,),
            )
            return model, name
        except Exception as e:  # keep fallback robust across timm versions
            last_error = e
    raise RuntimeError(
        f"Failed to create timm backbone {backbone_name!r}. Tried: {_candidate_backbone_names(backbone_name)}"
    ) from last_error


class SharedBackboneLiteProxyNet(nn.Module):
    """EfficientNet-Lite shared-backbone proxy model.

    Supported proxy tasks:
        - AE: token-wise local frame-stack reconstruction
        - SEP-direct: token-wise non-target feature prediction
        - CE / ArcFace: global embedding z
        - SimCLR / SimSiam: projection head on global embedding z

    Input:
        x: [B, 1, F, T], usually log-Mel [B, 1, 128, T]
    """

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_lite0.in1k",
        pretrained: bool = True,
        in_chans: int = 1,
        n_freq: int = 128,
        frame_stack: int = 5,
        num_classes: int = 10,
        feature_index: int = 3,
        token_time_mode: str = "upsample",
        token_hidden_dim: int = 128,
        projection_hidden_dim: int = 256,
        projection_dim: int = 128,
        simsiam_pred_hidden_dim: int = 256,
        normalize_projection: bool = False,
    ):
        super().__init__()

        if frame_stack % 2 != 1:
            raise ValueError("frame_stack should be odd, e.g. 5, 7, 9.")
        if token_time_mode not in {"native", "upsample"}:
            raise ValueError("token_time_mode must be 'native' or 'upsample'.")

        self.backbone_name_requested = backbone_name
        self.n_freq = int(n_freq)
        self.frame_stack = int(frame_stack)
        self.target_dim = int(n_freq) * int(frame_stack)
        self.token_time_mode = token_time_mode
        self.feature_index = int(feature_index)

        self.backbone, resolved_name = create_timm_feature_backbone(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            feature_index=feature_index,
        )
        self.backbone_name = resolved_name

        channels = self.backbone.feature_info.channels()
        if len(channels) != 1:
            raise RuntimeError(f"Expected one selected feature map, got channels={channels}")
        self.feat_dim = int(channels[0])

        self.ae_head = TokenMLP(
            in_dim=self.feat_dim,
            hidden_dim=token_hidden_dim,
            out_dim=self.target_dim,
            final_activation=None,
        )
        self.sep_direct_head = TokenMLP(
            in_dim=self.feat_dim,
            hidden_dim=token_hidden_dim,
            out_dim=self.target_dim,
            final_activation=None,
        )
        self.clf_head = nn.Linear(self.feat_dim, num_classes)
        self.projection_head = ProjectionHead(
            in_dim=self.feat_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
            normalize_output=normalize_projection,
        )
        self.simsiam_predictor = SimSiamPredictor(
            in_dim=projection_dim,
            hidden_dim=simsiam_pred_hidden_dim,
            out_dim=projection_dim,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"x should be [B, C, F, T], got {tuple(x.shape)}")

        input_frames = x.shape[-1]
        H = self.backbone(x)[0]      # [B, C, F', T']
        z = H.mean(dim=(2, 3))       # [B, C]

        tokens = H.mean(dim=2)       # [B, C, T']
        if self.token_time_mode == "upsample":
            tokens = F.interpolate(
                tokens,
                size=input_frames,
                mode="linear",
                align_corners=False,
            )
        tokens = tokens.transpose(1, 2).contiguous()  # [B, T_token, C]

        return {"H": H, "tokens": tokens, "z": z}

    def forward(self, x: torch.Tensor, task: str = "encode") -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        tokens = enc["tokens"]
        z = enc["z"]

        if task == "encode":
            return enc
        if task == "ae":
            enc["ae_pred"] = self.ae_head(tokens)
            return enc
        if task in {"sep", "sep_direct"}:
            enc["sep_pred"] = self.sep_direct_head(tokens)
            return enc
        if task in {"clf", "ce"}:
            enc["logits"] = self.clf_head(z)
            return enc
        if task in {"contrastive", "simclr", "simsiam_projection"}:
            enc["projection"] = self.projection_head(z)
            return enc
        if task == "simsiam_predict":
            projection = self.projection_head(z)
            enc["projection"] = projection
            enc["prediction"] = self.simsiam_predictor(projection)
            return enc
        if task == "all":
            enc["ae_pred"] = self.ae_head(tokens)
            enc["sep_pred"] = self.sep_direct_head(tokens)
            enc["logits"] = self.clf_head(z)
            enc["projection"] = self.projection_head(z)
            enc["prediction"] = self.simsiam_predictor(enc["projection"])
            return enc
        raise ValueError(f"Unknown task: {task}")
