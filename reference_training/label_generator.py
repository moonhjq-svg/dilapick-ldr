from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelConfig:
    window_length: int = 6000
    sampling_rate: float = 100.0
    gaussian_sigma_samples: float | None = None
    gaussian_sigma_seconds: float = 0.10
    detection_pre_samples: int = 100
    detection_post_samples: int = 500

    @property
    def sigma_samples(self) -> float:
        if self.gaussian_sigma_samples is not None:
            return float(self.gaussian_sigma_samples)
        return float(self.gaussian_sigma_seconds * self.sampling_rate)


def _valid_sample(sample: object, window_length: int) -> bool:
    try:
        value = int(sample)
    except (TypeError, ValueError):
        return False
    return 0 <= value < window_length


def gaussian_label(sample: object, config: LabelConfig | None = None) -> np.ndarray:
    config = config or LabelConfig()
    label = np.zeros(config.window_length, dtype=np.float32)
    if not _valid_sample(sample, config.window_length):
        return label
    center = int(sample)
    x = np.arange(config.window_length, dtype=np.float32)
    sigma = max(config.sigma_samples, 1e-6)
    label = np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)
    return np.clip(label, 0.0, 1.0)


def detection_label(
    p_sample: object,
    s_sample: object,
    config: LabelConfig | None = None,
    rule: str = "p_to_s_post",
) -> np.ndarray:
    config = config or LabelConfig()
    label = np.zeros(config.window_length, dtype=np.float32)
    if not (_valid_sample(p_sample, config.window_length) and _valid_sample(s_sample, config.window_length)):
        return label
    p = int(p_sample)
    s = int(s_sample)
    if rule == "p_to_s_post":
        start = max(0, p - config.detection_pre_samples)
        stop = min(config.window_length, s + config.detection_post_samples)
    elif rule == "around_p":
        start = max(0, p - config.detection_pre_samples)
        stop = min(config.window_length, p + config.detection_post_samples)
    else:
        raise ValueError(f"Unsupported detection window rule: {rule}")
    label[start:stop] = 1.0
    return label


def make_multitask_labels(
    p_sample: object,
    s_sample: object,
    trace_category: str = "earthquake_local",
    config: LabelConfig | None = None,
    detection_rule: str = "p_to_s_post",
) -> dict[str, np.ndarray]:
    config = config or LabelConfig()
    is_noise = str(trace_category).lower() == "noise"
    if is_noise:
        zero = np.zeros(config.window_length, dtype=np.float32)
        return {"detection": zero.copy(), "p": zero.copy(), "s": zero.copy()}
    return {
        "detection": detection_label(p_sample, s_sample, config, rule=detection_rule),
        "p": gaussian_label(p_sample, config),
        "s": gaussian_label(s_sample, config),
    }


def assert_label_smoke_checks() -> None:
    config = LabelConfig()
    event = make_multitask_labels(1200, 1800, "earthquake_local", config)
    noise = make_multitask_labels(-1, -1, "noise", config)
    assert event["p"].sum() > 0 and event["s"].sum() > 0
    assert np.all(noise["detection"] == 0) and np.all(noise["p"] == 0) and np.all(noise["s"] == 0)
    for labels in (event, noise):
        for curve in labels.values():
            assert curve.shape == (config.window_length,)
            assert float(curve.max()) <= 1.0
            assert np.isfinite(curve).all()


if __name__ == "__main__":
    assert_label_smoke_checks()
    print("label_generator_smoke=PASS")

