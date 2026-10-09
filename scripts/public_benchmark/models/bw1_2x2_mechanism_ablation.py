"""Frozen B-W1 2x2 mechanism ablation interface.

This module deliberately reuses every common B-W1 component.  The only
switches are the bottleneck TCN and the three task adapters.
"""
from __future__ import annotations

import torch
from torch import nn

from scripts.public_benchmark.models.ab_lightweight_joint_picker import (
    AdditiveUpsampleFusion1D,
    DepthwiseDownsample1D,
    DilatedTCNLongContext,
    HighResolutionDSResidualBlock,
    InvertedResidual1D,
    TaskBottleneckAdapter1D,
)


class BW1MechanismAblation(nn.Module):
    """The preregistered M00/M10/M01/M11 architecture family only."""

    def __init__(self, *, tcn_enabled: bool, adapter_enabled: bool) -> None:
        super().__init__()
        self.tcn_enabled = bool(tcn_enabled)
        self.adapter_enabled = bool(adapter_enabled)
        self.stem = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3, bias=False), nn.BatchNorm1d(16), nn.SiLU(inplace=False))
        self.encoder_high = HighResolutionDSResidualBlock(16)
        self.down1 = DepthwiseDownsample1D(16, 24)
        self.encoder_3000 = InvertedResidual1D(24)
        self.down2 = DepthwiseDownsample1D(24, 32)
        self.encoder_1500 = InvertedResidual1D(32)
        self.down3 = DepthwiseDownsample1D(32, 48)
        self.encoder_750 = InvertedResidual1D(48)
        self.long_context = DilatedTCNLongContext(48) if self.tcn_enabled else nn.Identity()
        self.up2 = AdditiveUpsampleFusion1D(48, 32, InvertedResidual1D(32))
        self.up1 = AdditiveUpsampleFusion1D(32, 24, InvertedResidual1D(24))
        self.up0 = AdditiveUpsampleFusion1D(24, 16, HighResolutionDSResidualBlock(16))
        if self.adapter_enabled:
            self.detection_adapter = TaskBottleneckAdapter1D()
            self.p_adapter = TaskBottleneckAdapter1D()
            self.s_adapter = TaskBottleneckAdapter1D()
        else:
            self.detection_adapter = nn.Conv1d(16, 1, 1, bias=True)
            self.p_adapter = nn.Conv1d(16, 1, 1, bias=True)
            self.s_adapter = nn.Conv1d(16, 1, 1, bias=True)

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
        if self.adapter_enabled:
            return {"detection_logits": self.detection_adapter(latent), "p_logits": self.p_adapter(latent), "s_logits": self.s_adapter(latent)}
        return {"detection_logits": self.detection_adapter(latent).squeeze(1), "p_logits": self.p_adapter(latent).squeeze(1), "s_logits": self.s_adapter(latent).squeeze(1)}
