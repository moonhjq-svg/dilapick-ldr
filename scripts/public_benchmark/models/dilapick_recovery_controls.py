"""Two fixed recovery controls for the 2026-09-16 validation-only study.

Frozen model files are imported, never edited. No test-driven architecture search.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.public_benchmark.models.dilapick_ldr import DILaPickLDR, _Projection1D
from scripts.public_benchmark.models.ab_lightweight_joint_picker import (
    HighResolutionDSResidualBlock,
    InvertedResidual1D,
)

SHARED = ('stem', 'encoder_high', 'down1', 'encoder_3000', 'down2',
          'encoder_1500', 'down3', 'encoder_750', 'long_context')


class LinearContextRecovery(DILaPickLDR):
    """Replace only deep-context subposition projection/shuffle by 3-logit interpolation.

The shallow P/S projection/shuffle, gamma values, and local correction retain
the original definitions AND same-seed initial tensors. This isolates the deep
recovery construction; it is not an all-linear decoder or a parameter match.
"""
    def __init__(self):
        super().__init__()
        self.context_projection = _Projection1D(48, 3)
        self.context_shuffle = nn.Identity()

    def forward(self, x):
        f2, f8 = self._features(x)
        context = F.interpolate(self.context_projection(f8), size=6000,
                                mode='linear', align_corners=False)
        detail = self.detail_shuffle(self.detail_projection(f2))
        logits = torch.cat((context[:, 0:1],
                            context[:, 1:2] + self.gamma_p * detail[:, 0:1],
                            context[:, 2:3] + self.gamma_s * detail[:, 1:2]), dim=1)
        return logits + self.local_calibrator(logits)


class _NarrowFusion(nn.Module):
    def __init__(self, in_channels, skip_channels, width, final=False):
        super().__init__()
        self.project = nn.Conv1d(in_channels, width, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(width)
        self.skip_project = nn.Conv1d(skip_channels, width, 1, bias=False)
        self.local_block = HighResolutionDSResidualBlock(width) if final else InvertedResidual1D(width)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        return self.local_block(self.project_bn(self.project(x)) + self.skip_project(skip))


class NarrowFeatureRecovery(nn.Module):
    """Progressive T/4 -> T/2 -> T feature recovery with one fixed shared width.

The width is locked by arithmetic proximity to LDR before any training. Width
is only exposed to the structural audit; the formal runner accepts the locked
value from protocol_lock.json and no validation-based override.
"""
    def __init__(self, width: int):
        super().__init__()
        if not isinstance(width, int) or width < 3:
            raise ValueError('Uniform feature width must be an integer >= 3.')
        base = DILaPickLDR()
        for name in SHARED:
            setattr(self, name, getattr(base, name))
        self.width = width
        self.up2 = _NarrowFusion(48, 32, width)
        self.up1 = _NarrowFusion(width, 24, width)
        self.up0 = _NarrowFusion(width, 16, width, final=True)
        self.head = nn.Conv1d(width, 3, 1, bias=True)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (3, 6000):
            raise ValueError(f'Expected [B,3,6000], got {tuple(x.shape)}')
        e0 = self.encoder_high(self.stem(x))
        e1 = self.encoder_3000(self.down1(e0))
        e2 = self.encoder_1500(self.down2(e1))
        e3 = self.encoder_750(self.down3(e2))
        z = self.up2(self.long_context(e3), e2)
        z = self.up1(z, e1)
        return self.head(self.up0(z, e0))


def make_control(name: str, width: int):
    if name == 'LINEAR_CONTEXT':
        return LinearContextRecovery()
    if name == 'NARROW_FEATURE':
        return NarrowFeatureRecovery(width)
    raise ValueError(f'Unknown locked control: {name}')
