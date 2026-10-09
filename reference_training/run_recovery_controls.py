"""Frozen, validation-only recovery comparison; preflight and six bounded runs.

Uses existing LDR loss, loaders, validation scorer, threshold selection, and
checkpoint ranking verbatim after hash verification. Never constructs test data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            f.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)


def source_preflight(root, lock):
    rows = []
    for rel, expected in lock['source_hashes'].items():
        path = root / rel
        actual = sha(path) if path.is_file() else None
        rows.append({'file': rel, 'expected': expected, 'actual': actual,
                     'passed': actual == expected})
    return rows


def load_frozen(root):
    sys.path.insert(0, str(root))
    os.environ['DILAPICK_LDR_ROOT'] = str(root)
    return importlib.import_module('scripts.public_benchmark.train_dilapick_ldr_seed42')


def preflight(root, base, lock, locksha):
    sources = source_preflight(root, lock)
    report = {'status': 'FAIL_PREFLIGHT', 'protocol_sha256': locksha,
              'source_checks': sources, 'test_waveforms_accessed': False,
              'training_started': False}
    write(base / 'server_preflight.json', report)
    assert all(x['passed'] for x in sources), 'Frozen source identity mismatch; do not repair by changing the lock after observing results.'
    frozen = load_frozen(root)
    import pandas as pd
    import torch
    from scripts.public_benchmark.audit_recovery_controls import cost
    from scripts.public_benchmark.models.dilapick_recovery_controls import make_control
    from scripts.public_benchmark.models.dilapick_ldr import DILaPickLDR
    frozen.INDEX = root / lock['data']['index']
    frozen.H5 = root / lock['data']['waveforms']
    assert sha(frozen.INDEX) == lock['data']['index_sha256'], 'Master split index mismatch'
    assert frozen.H5.is_file(), 'Waveform HDF5 is missing'
    index = pd.read_csv(frozen.INDEX)
    assert index.split.value_counts().to_dict() == {'train': 80000, 'val': 10000, 'test': 10000}
    val = index.loc[index.split.eq('val')].reset_index(drop=True)
    assert hashlib.sha256('\n'.join(val.trace_name.astype(str)).encode()).hexdigest() == lock['data']['validation_order_sha256']
    assert index.trace_name.is_unique, 'Duplicate trace identifier in the master split'
    assert (frozen.EPOCHS, frozen.BATCH_SIZE, frozen.WORKERS) == (20, 32, 2)
    assert list(frozen.GRID) == lock['training']['threshold_grid']
    frozen_gate = read(root / lock['references']['validation_gate'])
    assert sha(root / lock['references']['validation_gate']) == lock['references']['validation_gate_sha256']
    checkpoint_checks = []
    for seed in lock['seeds']:
        record = frozen_gate['per_seed'][str(seed)]
        checkpoint = Path(record['ldr_checkpoint'])
        if str(checkpoint).startswith('/home/u703/Hjq/'):
            checkpoint = root / str(checkpoint).removeprefix('/home/u703/Hjq/')
        assert sha(checkpoint) == record['ldr_checkpoint_sha256']
        raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = raw['model_state_dict']
        model = DILaPickLDR(); model.load_state_dict(state, strict=True)
        assert raw['seed'] == seed and raw['validation_metrics'] == record['validation']
        checkpoint_checks.append({'seed': seed, 'path': str(checkpoint), 'sha256': sha(checkpoint)})
    torch.set_num_threads(1)
    models = {}
    for variant in lock['variants']:
        model = make_control(variant, lock['narrow_width'])
        result = cost(model)
        expected = lock['complexity'][variant]
        assert result['parameters'] == expected['parameters'] and result['core_macs'] == expected['core_macs']
        model.train()
        logits = model(torch.randn(2, 3, 6000))
        loss = frozen.loss_fn(logits, torch.rand_like(logits)); loss.backward()
        assert torch.isfinite(logits).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        models[variant] = {k: result[k] for k in ['parameters', 'core_macs']}
    # Touch only training/validation records through the frozen loader.
    for split, count in [('train', 80000), ('val', 10000)]:
        ds = frozen.SteadMultitaskDataset(frozen.INDEX, frozen.H5, split=split)
        assert len(ds) == count
        sample = ds[0]
        assert tuple(sample['waveform'].shape) == (3, 6000)
        assert tuple(sample['labels'].shape) == (3, 6000)
        assert torch.isfinite(sample['waveform']).all()
        ds.close()
    report.update(status='PASS_RECOVERY_CONTROL_SERVER_PREFLIGHT', checkpoints=checkpoint_checks,
                  models=models, torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                  master_index_metadata_read=True, waveform_splits_read=['train', 'val'])
    write(base / 'server_preflight.json', report)
    return frozen, frozen_gate


def score_with_ids(frozen, model, loader, device):
    frame, loss = frozen.score(model, loader, device)
    metadata = loader.dataset.metadata.reset_index(drop=True)
    assert len(frame) == len(metadata)
    for column in ['trace_name', 'source_id', 'trace_category', 'sample_type', 'split']:
        if column in metadata.columns:
            frame[column] = metadata[column].to_numpy()
    assert frame.trace_name.is_unique and frame.split.eq('val').all()
    return frame, loss


def train_one(frozen, root, base, lock, locksha, variant, seed, reference):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.public_benchmark.models.dilapick_recovery_controls import make_control
    out = base / 'runs' / variant / f'seed{seed}'
    if out.exists():
        gate_path = out / 'completion_gate.json'
        if not gate_path.is_file():
            raise RuntimeError(f'Existing incomplete run requires inspection: {out}')
        gate = read(gate_path)
        assert gate['status'] == 'PASS_TRAINING_AND_VALIDATION_ARTIFACTS' and gate['protocol_sha256'] == locksha
        assert sha(out/'best_checkpoint.pt') == gate['checkpoint_sha256']
        assert sha(out/'validation_predictions.csv') == gate['prediction_sha256']
        assert sha(out/'metrics.json') == gate['metrics_sha256']
        assert sha(out/'epoch_history.csv') == gate['history_sha256']
        return read(out/'metrics.json')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'run_lock.json', {'protocol_sha256':locksha, 'variant':variant, 'seed':seed,
          'historical_LDR_retrained':False, 'test_waveforms_accessed':False}, exclusive=True)
    frozen.SEED = seed
    frozen.set_seed()
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    train_loader = frozen.loader(frozen.SteadMultitaskDataset(frozen.INDEX, frozen.H5, split='train'), True)
    val_loader = frozen.loader(frozen.SteadMultitaskDataset(frozen.INDEX, frozen.H5, split='val'), False)
    model = make_control(variant, lock['narrow_width']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9,.999), eps=1e-8,
                                 weight_decay=1e-4, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    best, history, start = None, [], time.time()
    for epoch in range(1, 21):
        model.train(); losses=[]; epoch_start=time.time()
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['waveform'].to(device, non_blocking=True))
            loss = frozen.loss_fn(logits, batch['labels'].to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gn):
                raise RuntimeError('Nonfinite gradient norm')
            optimizer.step(); losses.append(float(loss.detach().cpu()))
            if batch_index % 100 == 0:
                write(out/'progress.json', {'status':'TRAINING', 'epoch':epoch,
                      'batch':batch_index, 'batches':len(train_loader), 'elapsed_seconds':time.time()-start})
        frame, val_loss = score_with_ids(frozen, model, val_loader, device)
        thresholds = frozen.thresholds(frame)
        metrics = frozen.summary_metrics(frame, thresholds, val_loss)
        row = {'epoch':epoch, 'train_total_bce_loss':float(np.mean(losses)),
               'validation_total_bce_loss':val_loss, 'joint_f1':metrics['joint_f1'],
               'ps_mean_f1':metrics['ps_mean_f1'], 'detection_f1':metrics['detection']['f1'],
               'p_f1':metrics['p']['f1'], 's_f1':metrics['s']['f1'],
               'p_mae_seconds':metrics['p']['mae_seconds'], 's_mae_seconds':metrics['s']['mae_seconds'],
               'threshold_detection':thresholds['detection'], 'threshold_p':thresholds['p'],
               'threshold_s':thresholds['s'], 'epoch_seconds':time.time()-epoch_start}
        history.append(row)
        if best is None or frozen.ranking(metrics,epoch) > frozen.ranking(best['metrics'],best['epoch']):
            best={'epoch':epoch, 'metrics':metrics, 'thresholds':thresholds,
                  'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  'frame':frame.copy(deep=True)}
        scheduler.step()
        pd.DataFrame(history).to_csv(out/'epoch_history.csv',index=False,lineterminator='\n')
        print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
    # Re-load and re-score the selected weights before finalizing a run.
    model.load_state_dict(best['state'],strict=True)
    selected_frame, selected_loss = score_with_ids(frozen,model,val_loader,device)
    selected_metrics=frozen.summary_metrics(selected_frame,best['thresholds'],selected_loss)
    for task in ['detection','p','s']:
        assert selected_metrics[task] == best['metrics'][task], 'Selected-checkpoint score mismatch'
    assert selected_frame.trace_name.tolist() == best['frame'].trace_name.tolist()
    assert np.array_equal(selected_frame[['p_pred_sample','s_pred_sample']].to_numpy(),
                          best['frame'][['p_pred_sample','s_pred_sample']].to_numpy())
    checkpoint=out/'best_checkpoint.pt'
    torch.save({'model_state_dict':best['state'],'model':variant,'seed':seed,'best_epoch':best['epoch'],
                'validation_thresholds':best['thresholds'],'validation_metrics':best['metrics'],
                'protocol_sha256':locksha,'test_accessed':False},checkpoint)
    selected_frame.to_csv(out/'validation_predictions.csv',index=False,lineterminator='\n')
    result={'variant':variant,'seed':seed,'best_epoch':best['epoch'],'validation':best['metrics'],
            'thresholds':best['thresholds'],'complexity':lock['complexity'][variant],
            'ldr_reference_validation':reference['validation'],
            'checkpoint_sha256':sha(checkpoint),'protocol_sha256':locksha,
            'epochs_completed':20,'training_seconds':time.time()-start,'test_accessed':False}
    write(out/'metrics.json',result,exclusive=True)
    write(out/'completion_gate.json',{'status':'PASS_TRAINING_AND_VALIDATION_ARTIFACTS',
          'scientific_superiority':'NOT_IMPLIED_BY_COMPLETION', 'protocol_sha256':locksha,
          'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(out/'validation_predictions.csv'),
          'metrics_sha256':sha(out/'metrics.json'),'history_sha256':sha(out/'epoch_history.csv'),
          'epochs_completed':20,'test_accessed':False},exclusive=True)
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['preflight','train'])
    parser.add_argument('--root',type=Path,default=Path('/home/u703/Hjq'))
    parser.add_argument('--cuda-device',type=int,choices=[0,2,3])
    args=parser.parse_args()
    if args.action=='train' and args.cuda_device is None:
        parser.error('Choose one inspected physical GPU: --cuda-device 0, 2, or 3. GPU 1 is excluded.')
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.cuda_device)
    root=args.root.resolve()
    base=root/'experiments/public_benchmark_restart/recovery_controls_20260916'
    lockfile=base/'protocol_lock.json'
    lock=read(lockfile); locksha=sha(lockfile)
    assert lock['status']=='FROZEN_BEFORE_NEW_TRAINING'
    try:
        frozen,reference_gate=preflight(root,base,lock,locksha)
        if args.action=='preflight':
            print('PASS_RECOVERY_CONTROL_SERVER_PREFLIGHT'); return
        import torch
        assert torch.cuda.is_available(),'CUDA is required for formal training'
        results=[]
        for seed in lock['seeds']:
            for variant in lock['variants']:
                write(base/'scheduler_state.json',{'status':'RUNNING','variant':variant,'seed':seed,
                      'physical_gpu':args.cuda_device,'completed_runs':len(results),'total_runs':6,
                      'protocol_sha256':locksha,'pid':os.getpid()})
                result=train_one(frozen,root,base,lock,locksha,variant,seed,reference_gate['per_seed'][str(seed)])
                results.append(result)
        # Sources and original checkpoints must remain unchanged after all runs.
        assert all(x['passed'] for x in source_preflight(root,lock))
        for record in read(base/'server_preflight.json')['checkpoints']:
            assert sha(record['path'])==record['sha256']
        write(base/'training_completion_gate.json',{'status':'PASS_SIX_RECOVERY_CONTROL_VALIDATION_RUNS',
              'runs':len(results),'protocol_sha256':locksha,'test_accessed':False,
              'original_checkpoints_unchanged':True,'scientific_analysis_status':'PENDING_PAIRED_ANALYSIS_AND_LATENCY'})
        write(base/'scheduler_state.json',{'status':'TRAINING_COMPLETE_ANALYSIS_PENDING',
              'completed_runs':6,'protocol_sha256':locksha})
    except Exception:
        write(base/'execution_failure.json',{'status':'FAIL_EXECUTION','protocol_sha256':locksha,
              'traceback':traceback.format_exc(),'test_waveforms_accessed':False})
        raise


if __name__=='__main__':
    main()
