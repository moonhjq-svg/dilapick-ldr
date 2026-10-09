"""Pre-registered A/B Lightweight Joint Picker architecture lock.

The four public constructors differ only in bottleneck width (48 or 56) and
the 750-sample long-context module (single-layer BiGRU or seven-block TCN).
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class HighResolutionDSResidualBlock(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU(inplace=False)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.depthwise_bn(self.depthwise(x)))
        return self.activation(x + self.pointwise_bn(self.pointwise(y)))


class InvertedResidual1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = 2 * channels
        self.expand = nn.Conv1d(channels, expanded, 1, bias=False)
        self.expand_bn = nn.BatchNorm1d(expanded)
        self.depthwise = nn.Conv1d(expanded, expanded, 5, padding=2, groups=expanded, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(expanded)
        self.activation = nn.SiLU(inplace=False)
        self.project = nn.Conv1d(expanded, channels, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.expand_bn(self.expand(x)))
        y = self.activation(self.depthwise_bn(self.depthwise(y)))
        return self.activation(x + self.project_bn(self.project(y)))


class DepthwiseDownsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 5, stride=2, padding=2, groups=in_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(in_channels)
        self.activation = nn.SiLU(inplace=False)
        self.project = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.project_bn(self.project(self.activation(self.depthwise_bn(self.depthwise(x))))))


class AdditiveUpsampleFusion1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, local_block: nn.Module) -> None:
        super().__init__()
        self.project = nn.Conv1d(in_channels, skip_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(skip_channels)
        self.local_block = local_block

    def forward(self, decoder_feature: torch.Tensor, encoder_skip: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(decoder_feature, size=encoder_skip.shape[-1], mode="nearest")
        return self.local_block(self.project_bn(self.project(y)) + encoder_skip)


class BiGRULongContext(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gru = nn.GRU(channels, channels // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
        self.project = nn.Conv1d(channels, channels, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x.transpose(1, 2))
        y = self.project_bn(self.project(y.transpose(1, 2)))
        return self.activation(x + y)


class DilatedTCNResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, groups=channels, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU(inplace=False)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.depthwise_bn(self.depthwise(x)))
        return self.activation(x + self.pointwise_bn(self.pointwise(y)))


class DilatedTCNLongContext(nn.Module):
    DILATIONS = (1, 2, 4, 8, 16, 32, 64)

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(DilatedTCNResidualBlock(channels, d) for d in self.DILATIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TaskBottleneckAdapter1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv1d(16, 8, 1, bias=True)
        self.depthwise = nn.Conv1d(8, 8, 5, padding=2, groups=8, bias=True)
        self.activation = nn.SiLU(inplace=False)
        self.expand = nn.Conv1d(8, 16, 1, bias=True)
        self.logit = nn.Conv1d(16, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.depthwise(self.reduce(x)))
        return self.logit(self.activation(x + self.expand(y))).squeeze(1)


class LightweightJointPicker(nn.Module):
    """Common locked encoder/decoder with exactly one selectable context module."""
    def __init__(self, width: Literal[48, 56], context: Literal["bigru", "tcn"]) -> None:
        super().__init__()
        if width not in (48, 56) or context not in ("bigru", "tcn"):
            raise ValueError("only locked widths 48/56 and contexts bigru/tcn are permitted")
        self.width, self.context_kind = width, context
        self.stem = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3, bias=False), nn.BatchNorm1d(16), nn.SiLU(inplace=False))
        self.encoder_high = HighResolutionDSResidualBlock(16)
        self.down1 = DepthwiseDownsample1D(16, 24)
        self.encoder_3000 = InvertedResidual1D(24)
        self.down2 = DepthwiseDownsample1D(24, 32)
        self.encoder_1500 = InvertedResidual1D(32)
        self.down3 = DepthwiseDownsample1D(32, width)
        self.encoder_750 = InvertedResidual1D(width)
        self.long_context = BiGRULongContext(width) if context == "bigru" else DilatedTCNLongContext(width)
        self.up2 = AdditiveUpsampleFusion1D(width, 32, InvertedResidual1D(32))
        self.up1 = AdditiveUpsampleFusion1D(32, 24, InvertedResidual1D(24))
        self.up0 = AdditiveUpsampleFusion1D(24, 16, HighResolutionDSResidualBlock(16))
        self.detection_adapter = TaskBottleneckAdapter1D()
        self.p_adapter = TaskBottleneckAdapter1D()
        self.s_adapter = TaskBottleneckAdapter1D()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, 6000):
            raise ValueError(f"expected [B,3,6000], got {tuple(x.shape)}")
        e0 = self.encoder_high(self.stem(x))
        e1 = self.encoder_3000(self.down1(e0))
        e2 = self.encoder_1500(self.down2(e1))
        e3 = self.encoder_750(self.down3(e2))
        d2 = self.up2(self.long_context(e3), e2)
        d1 = self.up1(d2, e1)
        latent = self.up0(d1, e0)
        return {"detection_logits": self.detection_adapter(latent), "p_logits": self.p_adapter(latent), "s_logits": self.s_adapter(latent)}


class AW1(LightweightJointPicker):
    def __init__(self) -> None: super().__init__(48, "bigru")
class AW2(LightweightJointPicker):
    def __init__(self) -> None: super().__init__(56, "bigru")
class BW1(LightweightJointPicker):
    def __init__(self) -> None: super().__init__(48, "tcn")
class BW2(LightweightJointPicker):
    def __init__(self) -> None: super().__init__(56, "tcn")


MODELS = {"A-W1": AW1, "A-W2": AW2, "B-W1": BW1, "B-W2": BW2}
