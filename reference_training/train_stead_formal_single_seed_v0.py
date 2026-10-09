from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.public_benchmark.data.stead_multitask_dataset import SteadMultitaskDataset
from scripts.public_benchmark.models.losses import MultitaskBCEWithLogitsLoss
from scripts.public_benchmark.models.seislitenet_mt import SeisLiteNetMT, count_parameters

SEED = 20260630
INDEX = Path('/home/u703/Hjq/results/public_benchmark_restart/stead_50k_50k_index.csv')
H5 = Path('/home/u703/Hjq/datasets/STEAD_seisbench_mirror/waveforms.hdf5')
OUT = Path('/home/u703/Hjq/experiments/public_benchmark_restart/stead_formal_single_seed_v0_seed20260630')
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 2
SAMPLE_RATE_HZ = 100.0
PICK_TOLERANCE_SAMPLES = 50
FIXED_THRESHOLDS = {'detection': 0.5, 'p': 0.3, 's': 0.3}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def collate(batch):
    return {
        'waveform': torch.stack([x['waveform'] for x in batch], dim=0),
        'labels': torch.stack([x['labels'] for x in batch], dim=0),
        'metadata': [x['metadata'] for x in batch],
    }


def make_loader(split: str, shuffle: bool) -> DataLoader:
    ds = SteadMultitaskDataset(INDEX, H5, split=split)
    gen = torch.Generator().manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        generator=gen,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def finite_grads(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            return False
    return True


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def detection_metrics_from_arrays(det_max: np.ndarray, is_event: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = det_max >= threshold
    true = is_event.astype(bool)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    out = prf(tp, fp, fn)
    out['tn'] = tn
    return out


def phase_metrics_from_arrays(max_prob: np.ndarray, pred_peak: np.ndarray, true_peak: np.ndarray, true_valid: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    pred_pos = max_prob >= threshold
    has_true = true_valid.astype(bool)
    abs_err = np.abs(pred_peak - true_peak)
    within = abs_err <= PICK_TOLERANCE_SAMPLES
    tp_mask = pred_pos & has_true & within
    fp_mask = pred_pos & (~has_true | ~within)
    fn_mask = has_true & (~pred_pos | ~within)
    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    out: dict[str, float | int | None] = prf(tp, fp, fn)
    if tp:
        mae_samples = float(abs_err[tp_mask].mean())
        out['mae_samples'] = mae_samples
        out['mae_seconds'] = mae_samples / SAMPLE_RATE_HZ
    else:
        out['mae_samples'] = None
        out['mae_seconds'] = None
    out['tolerance_samples'] = PICK_TOLERANCE_SAMPLES
    out['tolerance_seconds'] = PICK_TOLERANCE_SAMPLES / SAMPLE_RATE_HZ
    return out


def metric_track(df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    is_event = df['is_event_label'].to_numpy(dtype=bool)
    return {
        'thresholds': thresholds,
        'detection': detection_metrics_from_arrays(df['detection_max_prob'].to_numpy(float), is_event, thresholds['detection']),
        'p': phase_metrics_from_arrays(
            df['p_max_prob'].to_numpy(float),
            df['p_pred_sample'].to_numpy(int),
            df['p_true_sample'].to_numpy(int),
            df['p_true_available'].to_numpy(bool),
            thresholds['p'],
        ),
        's': phase_metrics_from_arrays(
            df['s_max_prob'].to_numpy(float),
            df['s_pred_sample'].to_numpy(int),
            df['s_true_sample'].to_numpy(int),
            df['s_true_available'].to_numpy(bool),
            thresholds['s'],
        ),
    }


def select_thresholds_on_val(df: pd.DataFrame) -> dict[str, float]:
    det_grid = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
    phase_grid = [round(x, 2) for x in np.linspace(0.05, 0.9, 18)]
    selected: dict[str, float] = {}
    best_det = max(det_grid, key=lambda t: (metric_track(df, {'detection': t, 'p': 0.3, 's': 0.3})['detection']['f1'], t))
    selected['detection'] = float(best_det)
    best_p = max(phase_grid, key=lambda t: (metric_track(df, {'detection': 0.5, 'p': t, 's': 0.3})['p']['f1'], t))
    selected['p'] = float(best_p)
    best_s = max(phase_grid, key=lambda t: (metric_track(df, {'detection': 0.5, 'p': 0.3, 's': t})['s']['f1'], t))
    selected['s'] = float(best_s)
    return selected


def sample_loss(out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    det = F.binary_cross_entropy_with_logits(out['detection_logits'], y[:, 0, :], reduction='none').mean(dim=1)
    pp = F.binary_cross_entropy_with_logits(out['p_logits'], y[:, 1, :], reduction='none').mean(dim=1)
    ss = F.binary_cross_entropy_with_logits(out['s_logits'], y[:, 2, :], reduction='none').mean(dim=1)
    return det + pp + ss


def metadata_phase_sample(meta: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = meta.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
            return float(value)
        except TypeError:
            continue
    return None


def score(model, loader, loss_fn, device, score_csv: Path, thresholds_for_columns: dict[str, float] | None = None) -> dict[str, Any]:
    model.eval()
    start = time.time()
    losses, det_losses, p_losses, s_losses = [], [], [], []
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch['waveform'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            out = model(x)
            if not all(torch.isfinite(v).all().item() for v in out.values()):
                raise RuntimeError('non-finite model output during scoring')
            loss_dict = loss_fn(out, y)
            losses.append(float(loss_dict['total'].detach().cpu().item()))
            det_losses.append(float(loss_dict['detection'].detach().cpu().item()))
            p_losses.append(float(loss_dict['p'].detach().cpu().item()))
            s_losses.append(float(loss_dict['s'].detach().cpu().item()))
            per_sample_loss = sample_loss(out, y).detach().cpu().numpy()
            det_prob = torch.sigmoid(out['detection_logits']).detach().cpu()
            p_prob = torch.sigmoid(out['p_logits']).detach().cpu()
            s_prob = torch.sigmoid(out['s_logits']).detach().cpu()
            y_cpu = y.detach().cpu()
            for i, meta in enumerate(batch['metadata']):
                p_true = metadata_phase_sample(meta, ['p_arrival_sample', 'trace_p_arrival_sample'])
                s_true = metadata_phase_sample(meta, ['s_arrival_sample', 'trace_s_arrival_sample'])
                if p_true is None:
                    p_true = float(y_cpu[i, 1].argmax().item()) if y_cpu[i, 1].max().item() > 0 else -1.0
                if s_true is None:
                    s_true = float(y_cpu[i, 2].argmax().item()) if y_cpu[i, 2].max().item() > 0 else -1.0
                row = {
                    'trace_name': meta.get('trace_name'),
                    'split': meta.get('split'),
                    'trace_category': meta.get('trace_category'),
                    'sample_type': 'event' if y_cpu[i, 0].max().item() > 0 else 'noise',
                    'p_true_sample': int(round(p_true)) if p_true >= 0 else -1,
                    's_true_sample': int(round(s_true)) if s_true >= 0 else -1,
                    'p_true_available': int(p_true >= 0),
                    's_true_available': int(s_true >= 0),
                    'detection_max_prob': float(det_prob[i].max().item()),
                    'p_max_prob': float(p_prob[i].max().item()),
                    's_max_prob': float(s_prob[i].max().item()),
                    'p_pred_sample': int(p_prob[i].argmax().item()),
                    's_pred_sample': int(s_prob[i].argmax().item()),
                    'fixed_detection_decision': int(det_prob[i].max().item() >= FIXED_THRESHOLDS['detection']),
                    'fixed_p_decision': int(p_prob[i].max().item() >= FIXED_THRESHOLDS['p']),
                    'fixed_s_decision': int(s_prob[i].max().item() >= FIXED_THRESHOLDS['s']),
                    'is_event_label': int(y_cpu[i, 0].max().item() > 0),
                    'sample_loss_unweighted_bce_sum': float(per_sample_loss[i]),
                }
                if thresholds_for_columns is not None:
                    row['valselected_detection_decision'] = int(det_prob[i].max().item() >= thresholds_for_columns['detection'])
                    row['valselected_p_decision'] = int(p_prob[i].max().item() >= thresholds_for_columns['p'])
                    row['valselected_s_decision'] = int(s_prob[i].max().item() >= thresholds_for_columns['s'])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(score_csv, index=False)
    elapsed = time.time() - start
    fixed_track = metric_track(df, FIXED_THRESHOLDS)
    out_metrics: dict[str, Any] = {
        'mean_loss': float(np.mean(losses)),
        'component_mean_loss': {
            'detection': float(np.mean(det_losses)),
            'p': float(np.mean(p_losses)),
            's': float(np.mean(s_losses)),
        },
        'fixed_threshold_metrics': fixed_track,
        'num_samples': int(len(df)),
        'runtime_seconds': elapsed,
        'samples_per_second': len(df) / elapsed if elapsed else None,
    }
    if thresholds_for_columns is not None:
        out_metrics['validation_selected_metrics'] = metric_track(df, thresholds_for_columns)
    return {'metrics': out_metrics, 'scores': df}


def cuda_peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / 1024**2)


def shell_output(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        return f'unavailable: {exc}'


def write_preflight(device: torch.device) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    lines = [
        '# STEAD Formal Single-Seed V0 GPU Preflight',
        '',
        f'date: {shell_output(["date"])}',
        f'hostname: {shell_output(["hostname"])}',
        f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}',
        f'selected_device: {device}',
        f'torch_version: {torch.__version__}',
        f'cuda_available: {torch.cuda.is_available()}',
        f'cuda_device_name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}',
        f'gpu_free_bytes: {free_bytes}',
        f'gpu_total_bytes: {total_bytes}',
        f'batch_size: {BATCH_SIZE}',
        f'epochs: {EPOCHS}',
        'estimated_runtime: about 30-45 min from 1-epoch dry-run plus per-epoch validation scoring',
        '',
        '## nvidia-smi',
        shell_output(['nvidia-smi']),
    ]
    (OUT / 'gpu_preflight.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_report(metrics: dict[str, Any], history: list[dict[str, Any]], thresholds: dict[str, float], decision: str) -> None:
    lines = [
        '# STEAD Formal Single-Seed V0',
        '',
        f'Decision: `{decision}`',
        '',
        'This is the first formal single-seed STEAD run for SeisLiteNetMT on the deterministic 50k event + 50k noise subset. It is not a hyperparameter search, multi-seed validation, ablation, or SOTA claim.',
        '',
        '## Command',
        '',
        '`CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/u703/Hjq /home/u703/miniconda3/envs/huang/bin/python scripts/public_benchmark/train_stead_formal_single_seed_v0.py`',
        '',
        '## Protocol',
        '',
        f'- Seed: `{SEED}`',
        f'- Split sizes: `{metrics["subset_sizes"]}`',
        f'- Batch size: `{BATCH_SIZE}`',
        f'- Epochs: `{EPOCHS}`',
        f'- Optimizer: `AdamW`',
        f'- Learning rate: `{LR}`',
        '- Scheduler: `none`',
        '- Checkpoint selection: `best validation total loss`',
        f'- Fixed thresholds: `{FIXED_THRESHOLDS}`',
        f'- Validation-selected thresholds: `{thresholds}`',
        f'- Picking tolerance: `{PICK_TOLERANCE_SAMPLES}` samples = `{PICK_TOLERANCE_SAMPLES / SAMPLE_RATE_HZ}` s',
        '',
        '## Environment',
        '',
        f'- Device: `{metrics["device"]}`',
        f'- CUDA_VISIBLE_DEVICES: `{metrics["cuda_visible_devices"]}`',
        f'- CUDA device: `{metrics["cuda_device"]}`',
        f'- Torch: `{metrics["torch_version"]}`',
        f'- Model parameters: `{metrics["model_parameters"]}`',
        f'- Peak GPU memory MB: `{metrics["gpu_peak_memory_mb"]}`',
        '',
        '## Training History',
        '',
    ]
    for row in history:
        lines.append(f"- epoch `{row['epoch']}`: train_total=`{row['train_total_loss']}`, val_total=`{row['val_total_loss']}`, saved=`{row['checkpoint_saved']}`, samples/sec=`{row['samples_per_sec']}`")
    lines += [
        '',
        '## Primary Fixed-Threshold Metrics',
        '',
        f'- Validation: `{metrics["val_metrics"]["fixed_threshold_metrics"]}`',
        f'- Test: `{metrics["test_metrics"]["fixed_threshold_metrics"]}`',
        '',
        '## Secondary Validation-Selected Diagnostic Metrics',
        '',
        'Thresholds were selected on validation only and applied once to test.',
        '',
        f'- Validation: `{metrics["val_metrics"]["validation_selected_metrics"]}`',
        f'- Test: `{metrics["test_metrics"]["validation_selected_metrics"]}`',
        '',
        '## Loss And Runtime',
        '',
        f'- Best epoch: `{metrics["best_epoch"]}`',
        f'- Best validation loss: `{metrics["best_val_loss"]}`',
        f'- Final validation mean loss: `{metrics["val_metrics"]["mean_loss"]}`',
        f'- Test mean loss: `{metrics["test_metrics"]["mean_loss"]}`',
        f'- Training runtime seconds: `{metrics["training_runtime_seconds"]}`',
        f'- Scoring runtime seconds: `{metrics["scoring_runtime_seconds"]}`',
        f'- Overall runtime seconds: `{metrics["runtime_seconds"]}`',
        '',
        '## Checks',
        '',
        f'- Optimizer steps: `{metrics["optimizer_steps"]}`',
        f'- Checkpoint best saved/reloaded: `{metrics["checkpoint_reload_ok"]}`',
        f'- Checkpoint last saved: `{metrics["checkpoint_last_saved"]}`',
        f'- Gradients finite: `{metrics["gradient_finite_all"]}`',
        f'- Outputs finite: `{metrics["outputs_finite_all"]}`',
        f'- HDF5 failures: `{metrics["hdf5_failures"]}`',
        f'- Performance/SOTA claim allowed: `{metrics["performance_claim_allowed"]}`',
        '',
        '## Decision',
        '',
        f'`{decision}`',
    ]
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def maybe_make_figures(history: list[dict[str, Any]], val_scores: pd.DataFrame, test_scores: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = OUT / 'figures'
    fig_dir.mkdir(exist_ok=True)
    hist = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    plt.plot(hist['epoch'], hist['train_total_loss'], label='train total')
    plt.plot(hist['epoch'], hist['val_total_loss'], label='val total')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'loss_curve.png', dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    for label, data in [('val event', val_scores[val_scores['is_event_label'] == 1]), ('val noise', val_scores[val_scores['is_event_label'] == 0])]:
        plt.hist(data['detection_max_prob'], bins=40, alpha=0.55, label=label)
    plt.xlabel('detection max probability')
    plt.ylabel('count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'val_detection_score_histogram.png', dpi=160)
    plt.close()
    event_test = test_scores[test_scores['is_event_label'] == 1].copy()
    if len(event_test):
        event_test['p_error_samples'] = event_test['p_pred_sample'] - event_test['p_true_sample']
        event_test['s_error_samples'] = event_test['s_pred_sample'] - event_test['s_true_sample']
        plt.figure(figsize=(7, 4))
        plt.hist(event_test['p_error_samples'], bins=60, alpha=0.55, label='P')
        plt.hist(event_test['s_error_samples'], bins=60, alpha=0.55, label='S')
        plt.xlabel('pick error samples')
        plt.ylabel('count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'test_pick_error_histogram.png', dpi=160)
        plt.close()


def update_todo(decision: str) -> None:
    todo = Path('/home/u703/Hjq/results/public_benchmark_restart/real_stead_integration_todo.md')
    text = todo.read_text(encoding='utf-8') if todo.exists() else '# Real STEAD Integration TODO\n'
    entry = f"\n- [x] Formal single-seed STEAD V0: `{decision}`; output: `{OUT}`\n"
    if 'Formal single-seed STEAD V0' not in text:
        text = text.rstrip() + entry
    else:
        lines = []
        for line in text.splitlines():
            if 'Formal single-seed STEAD V0' in line:
                lines.append(entry.strip())
            else:
                lines.append(line)
        text = '\n'.join(lines) + '\n'
    todo.write_text(text, encoding='utf-8')


def main() -> None:
    start_all = time.time()
    if OUT.exists():
        existing = {p.name for p in OUT.iterdir()}
        allowed = {'run.log', 'gpu_preflight.txt'}
        if existing - allowed:
            raise SystemExit(f'Output directory exists and has protected files: {OUT} {sorted(existing)}')
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('GPU_BLOCKED: CUDA is not available')
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 8 * 1024**3:
        write_preflight(device)
        raise SystemExit(f'GPU_BLOCKED: less than 8GB free GPU memory ({free_bytes} bytes)')
    torch.cuda.reset_peak_memory_stats()
    write_preflight(device)

    train_loader = make_loader('train', True)
    val_loader = make_loader('val', False)
    test_loader = make_loader('test', False)
    model = SeisLiteNetMT().to(device)
    loss_fn = MultitaskBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history: list[dict[str, Any]] = []
    best_val_loss = float('inf')
    best_epoch = -1
    optimizer_steps = 0
    grad_finite_all = True
    outputs_finite_all = True
    hdf5_failures = 0
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        losses, det_losses, p_losses, s_losses = [], [], [], []
        num_samples = 0
        for batch in train_loader:
            x = batch['waveform'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            num_samples += int(x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            outputs_finite_all = outputs_finite_all and all(torch.isfinite(v).all().item() for v in out.values())
            loss_dict = loss_fn(out, y)
            loss = loss_dict['total']
            if not torch.isfinite(loss).item():
                raise RuntimeError('non-finite training loss')
            loss.backward()
            grad_finite_all = grad_finite_all and finite_grads(model)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu().item()))
            det_losses.append(float(loss_dict['detection'].detach().cpu().item()))
            p_losses.append(float(loss_dict['p'].detach().cpu().item()))
            s_losses.append(float(loss_dict['s'].detach().cpu().item()))
        epoch_elapsed = time.time() - epoch_start
        val_epoch = score(model, val_loader, loss_fn, device, OUT / f'val_scores_epoch{epoch}.csv')
        val_metrics = val_epoch['metrics']
        checkpoint_saved = False
        if val_metrics['mean_loss'] < best_val_loss:
            best_val_loss = val_metrics['mean_loss']
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'seed': SEED,
                'model_parameters': count_parameters(model),
                'fixed_thresholds': FIXED_THRESHOLDS,
                'selection_rule': 'best validation total loss',
            }, OUT / 'checkpoint_best.pt')
            checkpoint_saved = True
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_metrics['mean_loss'],
            'seed': SEED,
            'model_parameters': count_parameters(model),
        }, OUT / 'checkpoint_last.pt')
        row = {
            'epoch': epoch,
            'train_total_loss': float(np.mean(losses)),
            'train_detection_loss': float(np.mean(det_losses)),
            'train_p_loss': float(np.mean(p_losses)),
            'train_s_loss': float(np.mean(s_losses)),
            'val_total_loss': val_metrics['mean_loss'],
            'val_detection_loss': val_metrics['component_mean_loss']['detection'],
            'val_p_loss': val_metrics['component_mean_loss']['p'],
            'val_s_loss': val_metrics['component_mean_loss']['s'],
            'epoch_runtime_sec': epoch_elapsed,
            'samples_per_sec': num_samples / epoch_elapsed if epoch_elapsed else None,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'checkpoint_saved': int(checkpoint_saved),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'training_history.csv', index=False)
        print(json.dumps(row), flush=True)
    training_runtime = time.time() - training_start

    checkpoint = torch.load(OUT / 'checkpoint_best.pt', map_location=device)
    best_model = SeisLiteNetMT().to(device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        reload_out = best_model(batch['waveform'].to(device, non_blocking=True))
        reload_ok = all(torch.isfinite(v).all().item() for v in reload_out.values())

    val_initial = score(best_model, val_loader, loss_fn, device, OUT / 'val_scores.csv')
    selected_thresholds = select_thresholds_on_val(val_initial['scores'])
    val_scores = val_initial['scores']
    val_scores['valselected_detection_decision'] = (val_scores['detection_max_prob'] >= selected_thresholds['detection']).astype(int)
    val_scores['valselected_p_decision'] = (val_scores['p_max_prob'] >= selected_thresholds['p']).astype(int)
    val_scores['valselected_s_decision'] = (val_scores['s_max_prob'] >= selected_thresholds['s']).astype(int)
    val_scores.to_csv(OUT / 'val_scores.csv', index=False)
    val_metrics = val_initial['metrics']
    val_metrics['validation_selected_metrics'] = metric_track(val_scores, selected_thresholds)
    test_final = score(best_model, test_loader, loss_fn, device, OUT / 'test_scores.csv', selected_thresholds)
    test_scores = test_final['scores']
    test_metrics = test_final['metrics']
    scoring_runtime = val_metrics['runtime_seconds'] + test_metrics['runtime_seconds']
    runtime = time.time() - start_all

    fixed = test_metrics['fixed_threshold_metrics']
    selected = test_metrics['validation_selected_metrics']
    non_degenerate = (
        fixed['detection']['tp'] > 0 and fixed['detection']['f1'] > 0 and
        fixed['p']['tp'] > 0 and fixed['s']['tp'] > 0 and
        selected['detection']['tp'] > 0 and selected['p']['tp'] > 0 and selected['s']['tp'] > 0
    )
    loss_decreased = history[-1]['train_total_loss'] < history[0]['train_total_loss']
    all_checks = reload_ok and grad_finite_all and outputs_finite_all and hdf5_failures == 0
    if all_checks and non_degenerate and loss_decreased:
        decision = 'PASS_READY_FOR_BASELINE_EVAL'
    elif all_checks:
        decision = 'PASS_NEEDS_DIAGNOSIS'
    else:
        decision = 'FAIL'

    metrics = {
        'decision': decision,
        'formal_training': True,
        'performance_claim_allowed': False,
        'sota_claim_allowed': False,
        'seed': SEED,
        'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_device': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'subset_sizes': {'train': len(train_loader.dataset), 'val': len(val_loader.dataset), 'test': len(test_loader.dataset)},
        'batch_size': BATCH_SIZE,
        'epoch_count': EPOCHS,
        'learning_rate': LR,
        'optimizer': 'AdamW',
        'scheduler': None,
        'num_workers': NUM_WORKERS,
        'selection_rule': 'best checkpoint selected by lowest validation total loss',
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'model_parameters': count_parameters(best_model),
        'optimizer_steps': optimizer_steps,
        'gradient_finite_all': grad_finite_all,
        'outputs_finite_all': outputs_finite_all,
        'checkpoint_reload_ok': reload_ok,
        'checkpoint_last_saved': (OUT / 'checkpoint_last.pt').exists(),
        'checkpoint_best_path': str(OUT / 'checkpoint_best.pt'),
        'checkpoint_last_path': str(OUT / 'checkpoint_last.pt'),
        'fixed_thresholds': FIXED_THRESHOLDS,
        'validation_selected_thresholds': selected_thresholds,
        'threshold_selection': 'selected on validation split only; applied to test once',
        'picking_tolerance_samples': PICK_TOLERANCE_SAMPLES,
        'picking_tolerance_seconds': PICK_TOLERANCE_SAMPLES / SAMPLE_RATE_HZ,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_runtime_seconds': training_runtime,
        'scoring_runtime_seconds': scoring_runtime,
        'runtime_seconds': runtime,
        'gpu_peak_memory_mb': cuda_peak_mb(),
        'hdf5_failures': hdf5_failures,
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    maybe_make_figures(history, val_scores, test_scores)
    write_report(metrics, history, selected_thresholds, decision)
    if decision == 'PASS_READY_FOR_BASELINE_EVAL':
        update_todo(decision)
    print(json.dumps({
        'decision': decision,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_fixed_detection_f1': fixed['detection']['f1'],
        'test_fixed_p_f1': fixed['p']['f1'],
        'test_fixed_s_f1': fixed['s']['f1'],
        'test_valselected_detection_f1': selected['detection']['f1'],
        'test_valselected_p_f1': selected['p']['f1'],
        'test_valselected_s_f1': selected['s']['f1'],
        'runtime_seconds': runtime,
        'gpu_peak_memory_mb': metrics['gpu_peak_memory_mb'],
    }, indent=2), flush=True)
    if decision == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
