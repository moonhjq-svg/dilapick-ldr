"""Validation-only seed-42 run for DILaPick-LDR.

The loss, optimizer, schedule, data split, batch size, epochs, and validation
threshold grid are inherited unchanged from the frozen DILaPick protocol.
The test dataset is deliberately never constructed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("DILAPICK_LDR_ROOT", "/home/u703/Hjq"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.public_benchmark.data.stead_multitask_dataset import SteadMultitaskDataset
from scripts.public_benchmark.models.dilapick_ldr import DILaPickLDR
from scripts.public_benchmark.train_ab_lightweight_joint_picker_stage_b import (
    GRID,
    H5,
    INDEX,
    metric_track,
    thresholds,
)

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
WORKERS = 2
OUT = ROOT / "experiments/public_benchmark_restart/dilapick_ldr/seed42"
PREFLIGHT = ROOT / "experiments/public_benchmark_restart/dilapick_ldr/pretraining_audit.json"
BASELINE_METRICS = ROOT / "experiments/public_benchmark_restart/m10_final_candidate_seed42/metrics_full_precision.json"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([row["waveform"] for row in rows]),
        "labels": torch.stack([row["labels"] for row in rows]),
        "metadata": [row["metadata"] for row in rows],
    }


def loader(dataset: SteadMultitaskDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=WORKERS,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
        persistent_workers=True,
    )


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Same three-head mean BCEWithLogits as frozen DILaPick.
    losses = tuple(nn.functional.binary_cross_entropy_with_logits(logits[:, index, :], labels[:, index, :], reduction="mean") for index in range(3))
    return sum(losses) / 3


def phase_value(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key, metadata.get(f"trace_{key}", -1))
    return int(value) if pd.notna(value) and 0 <= float(value) < 6000 else -1


def score(model: nn.Module, data: DataLoader, device: torch.device) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in data:
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(waveforms)
            if tuple(logits.shape[1:]) != (3, 6000) or not torch.isfinite(logits).all():
                raise RuntimeError("invalid validation logits")
            losses.append(float(loss_fn(logits, labels).cpu()))
            probabilities = torch.sigmoid(logits).cpu()
            for index, metadata in enumerate(batch["metadata"]):
                p_true = phase_value(metadata, "p_arrival_sample")
                s_true = phase_value(metadata, "s_arrival_sample")
                rows.append(
                    {
                        "is_event_label": int(labels[index, 0].max().item() > 0),
                        "detection_max_prob": float(probabilities[index, 0].max()),
                        "p_max_prob": float(probabilities[index, 1].max()),
                        "s_max_prob": float(probabilities[index, 2].max()),
                        "p_pred_sample": int(probabilities[index, 1].argmax()),
                        "s_pred_sample": int(probabilities[index, 2].argmax()),
                        "p_true_sample": p_true,
                        "s_true_sample": s_true,
                        "p_true_available": int(p_true >= 0),
                        "s_true_available": int(s_true >= 0),
                    }
                )
    return pd.DataFrame(rows), float(np.mean(losses))


def summary_metrics(frame: pd.DataFrame, selected: dict[str, float], validation_loss: float) -> dict[str, Any]:
    metrics = metric_track(frame, selected)
    metrics["validation_total_bce_loss"] = validation_loss
    metrics["joint_f1"] = (metrics["detection"]["f1"] + metrics["p"]["f1"] + metrics["s"]["f1"]) / 3
    metrics["ps_mean_f1"] = (metrics["p"]["f1"] + metrics["s"]["f1"]) / 2
    return metrics


def ranking(metrics: dict[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        metrics["joint_f1"],
        metrics["ps_mean_f1"],
        metrics["s"]["recall"],
        -metrics["validation_total_bce_loss"],
        -epoch,
    )


def preflight() -> dict[str, Any]:
    gate = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if gate["status"] != "PASS_DILAPICK_LDR_PRETRAIN_MAC_GATE" or not gate["training_authorized"]:
        raise RuntimeError("DILaPick-LDR MAC/parameter hard gate failed")
    if not BASELINE_METRICS.exists():
        raise RuntimeError(f"frozen seed-42 baseline metrics missing: {BASELINE_METRICS}")
    if len(SteadMultitaskDataset(INDEX, H5, split="train")) != 80_000:
        raise RuntimeError("frozen train split cardinality mismatch")
    if len(SteadMultitaskDataset(INDEX, H5, split="val")) != 10_000:
        raise RuntimeError("frozen validation split cardinality mismatch")
    return gate


def train(gate: dict[str, Any]) -> dict[str, Any]:
    if OUT.exists():
        raise RuntimeError(f"refuse overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = loader(SteadMultitaskDataset(INDEX, H5, split="train"), True)
    validation_loader = loader(SteadMultitaskDataset(INDEX, H5, split="val"), False)
    model = DILaPickLDR().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["waveform"].to(device, non_blocking=True))
            total = loss_fn(logits, batch["labels"].to(device, non_blocking=True))
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss epoch={epoch} batch={batch_index}")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient norm epoch={epoch} batch={batch_index}")
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        frame, validation_loss = score(model, validation_loader, device)
        selected = thresholds(frame)
        metrics = summary_metrics(frame, selected, validation_loss)
        row = {
            "epoch": epoch,
            "train_total_bce_loss": float(np.mean(losses)),
            "validation_total_bce_loss": validation_loss,
            "joint_f1": metrics["joint_f1"],
            "ps_mean_f1": metrics["ps_mean_f1"],
            "detection_f1": metrics["detection"]["f1"],
            "p_f1": metrics["p"]["f1"],
            "s_f1": metrics["s"]["f1"],
            "p_mae_seconds": metrics["p"]["mae_seconds"],
            "s_mae_seconds": metrics["s"]["mae_seconds"],
            "threshold_detection": selected["detection"],
            "threshold_p": selected["p"],
            "threshold_s": selected["s"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        if best is None or ranking(metrics, epoch) > ranking(best["metrics"], best["epoch"]):
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "thresholds": selected,
                "state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            }
        scheduler.step()
        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False, lineterminator="\n")
        print(json.dumps({"epoch": epoch, "ps_mean_f1": metrics["ps_mean_f1"], "detection_f1": metrics["detection"]["f1"], "p_mae_seconds": metrics["p"]["mae_seconds"], "s_mae_seconds": metrics["s"]["mae_seconds"]}), flush=True)
    assert best is not None
    checkpoint = OUT / "best_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best["state"],
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_epoch": best["epoch"],
            "validation_thresholds": best["thresholds"],
            "validation_metrics": best["metrics"],
            "model": "DILaPick-LDR",
            "seed": SEED,
            "test_accessed": False,
        },
        checkpoint,
    )
    return {
        "status": "COMPLETE_STEAD_VALIDATION_ONLY",
        "model": "DILaPick-LDR",
        "seed": SEED,
        "best_epoch": best["epoch"],
        "validation": best["metrics"],
        "thresholds": best["thresholds"],
        "parameters": int(sum(parameter.numel() for parameter in DILaPickLDR().parameters())),
        "macs": gate["models"][1]["estimated_macs"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha_file(checkpoint),
        "training_seconds": time.time() - started,
        "complete_epochs": EPOCHS,
        "test_accessed": False,
    }


def write_report(gate: dict[str, Any], result: dict[str, Any]) -> None:
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))["result"]["validation"]
    current = result["validation"]
    comparisons = {
        "ps_mean_f1_delta_ldr_minus_frozen": current["ps_mean_f1"] - baseline["ps_mean_f1"],
        "detection_f1_delta_ldr_minus_frozen": current["detection"]["f1"] - baseline["detection"]["f1"],
        "p_mae_seconds_delta_ldr_minus_frozen": current["p"]["mae_seconds"] - baseline["p"]["mae_seconds"],
        "s_mae_seconds_delta_ldr_minus_frozen": current["s"]["mae_seconds"] - baseline["s"]["mae_seconds"],
    }
    checks = {
        "ps_mean_f1": comparisons["ps_mean_f1_delta_ldr_minus_frozen"] >= -0.003,
        "detection_f1": comparisons["detection_f1_delta_ldr_minus_frozen"] >= -0.001,
        "p_mae_seconds": comparisons["p_mae_seconds_delta_ldr_minus_frozen"] <= 0.01,
        "s_mae_seconds": comparisons["s_mae_seconds_delta_ldr_minus_frozen"] <= 0.01,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS_DILAPICK_LDR_SEED42_VALIDATION_GATE" if passed else "FAIL_DILAPICK_LDR_SEED42_VALIDATION_GATE",
        "test_accessed": False,
        "baseline_metrics_path": str(BASELINE_METRICS),
        "baseline_metrics_sha256": sha_file(BASELINE_METRICS),
        "baseline_validation": baseline,
        "ldr_validation": current,
        "comparisons": comparisons,
        "limits": {"ps_mean_f1_drop": 0.003, "detection_f1_drop": 0.001, "p_mae_seconds_worsening": 0.01, "s_mae_seconds_worsening": 0.01},
        "checks": checks,
        "result": result,
    }
    write_json(OUT / "seed42_validation_gate.json", payload)
    report = [
        f"# {'PASS' if passed else 'FAIL'} DILaPick-LDR seed 42 validation gate",
        "",
        "本报告仅使用冻结 train/validation split；未构造、读取或生成 test 数据、test 阈值和 test 结果。",
        "",
        "| 指标 | 冻结 DILaPick | LDR | LDR - baseline | 门槛 | 结果 |",
        "|---|---:|---:|---:|---:|---|",
        f"| P/S mean F1 | {baseline['ps_mean_f1']:.9f} | {current['ps_mean_f1']:.9f} | {comparisons['ps_mean_f1_delta_ldr_minus_frozen']:+.9f} | >= -0.003 | {'PASS' if checks['ps_mean_f1'] else 'FAIL'} |",
        f"| Detection F1 | {baseline['detection']['f1']:.9f} | {current['detection']['f1']:.9f} | {comparisons['detection_f1_delta_ldr_minus_frozen']:+.9f} | >= -0.001 | {'PASS' if checks['detection_f1'] else 'FAIL'} |",
        f"| P MAE (s) | {baseline['p']['mae_seconds']:.9f} | {current['p']['mae_seconds']:.9f} | {comparisons['p_mae_seconds_delta_ldr_minus_frozen']:+.9f} | <= +0.010 | {'PASS' if checks['p_mae_seconds'] else 'FAIL'} |",
        f"| S MAE (s) | {baseline['s']['mae_seconds']:.9f} | {current['s']['mae_seconds']:.9f} | {comparisons['s_mae_seconds_delta_ldr_minus_frozen']:+.9f} | <= +0.010 | {'PASS' if checks['s_mae_seconds'] else 'FAIL'} |",
        "",
        f"- Best epoch: {result['best_epoch']}",
        f"- LDR parameters: {result['parameters']:,}",
        f"- LDR total MACs: {result['macs']:,}",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Overall conclusion: **{'PASS' if passed else 'FAIL'}**",
    ]
    (OUT / "seed42_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "completion_gate.json", {"status": payload["status"], "test_accessed": False, "formal_attempts": 1})


def main() -> None:
    gate = preflight()
    result = train(gate)
    write_report(gate, result)
    print(json.dumps({"status": "COMPLETE", "validation_gate": str(OUT / "seed42_validation_gate.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
