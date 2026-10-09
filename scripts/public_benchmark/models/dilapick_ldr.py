"""DILaPick-LDR: logit-domain reconstruction decoder.

The encoder and seven-block dilated TCN are the frozen DILaPick components.
Only the shared feature-recovery decoder and task heads are replaced.
"""
from __future__ import annotations

import torch
from torch import nn

from scripts.public_benchmark.models.ab_lightweight_joint_picker import (
    DepthwiseDownsample1D,
    DilatedTCNLongContext,
    HighResolutionDSResidualBlock,
    InvertedResidual1D,
)


class TemporalShuffle1D(nn.Module):
    """Rearrange channel groups into the temporal axis.

    ``C_in = C_out * upscale_factor`` and the operation is parameter-free:
    ``[B,C_in,L] -> [B,C_out,L*upscale_factor]``.
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        if not isinstance(upscale_factor, int) or upscale_factor < 1:
            raise ValueError("upscale_factor must be a positive integer")
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,C,T], got {tuple(x.shape)}")
        batch, channels, length = x.shape
        factor = self.upscale_factor
        if channels % factor != 0:
            raise ValueError(f"input channels {channels} must be divisible by {factor}")
        return (
            x.reshape(batch, channels // factor, factor, length)
            .permute(0, 1, 3, 2)
            .reshape(batch, channels // factor, length * factor)
        )


class _Projection1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.normalization = nn.BatchNorm1d(in_channels)
        self.activation = nn.SiLU(inplace=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.normalization(self.depthwise(x))))


class DILaPickLDR(nn.Module):
    """DILaPick with logit-domain reconstruction.

    ``forward`` returns a single logits tensor with shape ``[B,3,6000]`` in
    D/P/S order. ``as_task_dict`` provides the old evaluator's named-logit
    view without changing the logits, loss, or post-processing definitions.
    """

    def __init__(self) -> None:
        super().__init__()
        # Frozen DILaPick encoder.
        self.stem = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(inplace=False),
        )
        self.encoder_high = HighResolutionDSResidualBlock(16)
        self.down1 = DepthwiseDownsample1D(16, 24)
        self.encoder_3000 = InvertedResidual1D(24)
        self.down2 = DepthwiseDownsample1D(24, 32)
        self.encoder_1500 = InvertedResidual1D(32)
        self.down3 = DepthwiseDownsample1D(32, 48)
        self.encoder_750 = InvertedResidual1D(48)
        self.long_context = DilatedTCNLongContext(48)

        # LDR decoder: no feature upsampling, high-resolution skip concat, or
        # wide feature recovery blocks.
        self.context_projection = _Projection1D(48, 24)
        self.context_shuffle = TemporalShuffle1D(8)
        self.detail_projection = _Projection1D(24, 4)
        self.detail_shuffle = TemporalShuffle1D(2)
        self.gamma_p = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.gamma_s = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.local_calibrator = nn.Conv1d(3, 3, kernel_size=9, padding=4, groups=3, bias=True)

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, 6000):
            raise ValueError(f"expected [B,3,6000], got {tuple(x.shape)}")
        e0 = self.encoder_high(self.stem(x))
        f2 = self.encoder_3000(self.down1(e0))
        e2 = self.encoder_1500(self.down2(f2))
        f8 = self.encoder_750(self.down3(e2))
        return f2, self.long_context(f8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f8 = self._features(x)
        context = self.context_shuffle(self.context_projection(f8))
        detail = self.detail_shuffle(self.detail_projection(f2))
        logits = torch.cat(
            (
                context[:, 0:1, :],
                context[:, 1:2, :] + self.gamma_p * detail[:, 0:1, :],
                context[:, 2:3, :] + self.gamma_s * detail[:, 1:2, :],
            ),
            dim=1,
        )
        return logits + self.local_calibrator(logits)

    @staticmethod
    def as_task_dict(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 3 or tuple(logits.shape[1:]) != (3, 6000):
            raise ValueError(f"expected logits [B,3,6000], got {tuple(logits.shape)}")
        return {
            "detection_logits": logits[:, 0, :],
            "p_logits": logits[:, 1, :],
            "s_logits": logits[:, 2, :],
        }


__all__ = ["DILaPickLDR", "TemporalShuffle1D"]
