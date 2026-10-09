"""Frozen Stage B runner: exactly A-W1, A-W2, B-W1, B-W2; never loads test."""
from __future__ import annotations

import hashlib, json, random, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

LOCAL_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_ROOT) not in sys.path: sys.path.insert(0, str(LOCAL_ROOT))
from scripts.public_benchmark.data.stead_multitask_dataset import SteadMultitaskDataset
from scripts.public_benchmark.models.ab_lightweight_joint_picker import MODELS
from scripts.public_benchmark.train_stead_formal_single_seed_v0 import metric_track

ROOT = Path('/home/u703/Hjq')
INDEX = ROOT/'results/public_benchmark_restart/stead_50k_50k_index.csv'
H5 = ROOT/'datasets/STEAD_seisbench_mirror/waveforms.hdf5'
OUT = ROOT/'experiments/public_benchmark_restart/ab_lightweight_joint_picker_stage_b'
STAGE_A = ROOT/'experiments/public_benchmark_restart/ab_lightweight_joint_picker_stage_a/stage_a_metrics.json'
SEED, EPOCHS, BATCH, WORKERS = 41, 15, 32, 2
SALT = 'AB_LIGHTWEIGHT_STAGE_B_V1_SEED41'
GRID = [round(x, 2) for x in np.arange(.05, 1.0, .05)]
TOLERANCE_SAMPLES = 50
ORDER = ('A-W1','A-W2','B-W1','B-W2')

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def sha_text(value: str) -> str: return hashlib.sha256(value.encode('utf-8')).hexdigest()
def write_json(path: Path, value: Any) -> None: path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
def collate(rows): return {'waveform':torch.stack([r['waveform'] for r in rows]),'labels':torch.stack([r['labels'] for r in rows]),'metadata':[r['metadata'] for r in rows]}
def loader(ds: SteadMultitaskDataset, shuffle: bool) -> DataLoader:
    return DataLoader(ds,batch_size=BATCH,shuffle=shuffle,drop_last=False,num_workers=WORKERS,collate_fn=collate,generator=torch.Generator().manual_seed(SEED),pin_memory=True,persistent_workers=True)
def valid_phase(s: pd.Series) -> pd.Series: return s.notna() & (s >= 0) & (s < 6000)

def build_subset() -> tuple[Path, dict[str,Any]]:
    metadata=pd.read_csv(INDEX)
    train=metadata[metadata['split'].eq('train')].copy()
    event=train[train['sample_type'].eq('event') & valid_phase(train['trace_p_arrival_sample']) & valid_phase(train['trace_s_arrival_sample'])].copy()
    noise=train[train['sample_type'].eq('noise')].copy()
    if len(event)<10000 or len(noise)<10000: raise RuntimeError(f'insufficient eligible train pools event={len(event)} noise={len(noise)}')
    for frame, kind in ((event,'event'),(noise,'noise')):
        frame['selection_hash']=frame['trace_name'].astype(str).map(lambda x:sha_text(x+SALT)); frame['sample_type']=kind
    selected=pd.concat([event.sort_values('selection_hash',kind='mergesort').head(10000),noise.sort_values('selection_hash',kind='mergesort').head(10000)],ignore_index=True)
    selected['merged_order_hash']=selected['trace_name'].astype(str).map(lambda x:sha_text(x+SALT+'_merged'))
    selected=selected.sort_values('merged_order_hash',kind='mergesort').reset_index(drop=True)
    manifest=selected.assign(sample_id=selected.trace_name,p_label_valid=valid_phase(selected.trace_p_arrival_sample),s_label_valid=valid_phase(selected.trace_s_arrival_sample),source_split='train')[['sample_id','sample_type','source_split','p_label_valid','s_label_valid','selection_hash','merged_order_hash']]
    path=OUT/'stage_b_train_subset.csv'; manifest.to_csv(path,index=False,encoding='utf-8',lineterminator='\n')
    lock={'salt':SALT,'event_count':10000,'noise_count':10000,'total_count':20000,'columns':list(manifest.columns),'selection_rule':'SHA256(sample_id + fixed_salt), first 10000 per eligible event/noise pool; merged SHA256(sample_id + fixed_salt + _merged)','sha256':sha_file(path),'sample_id_order_sha256':sha_text('\n'.join(manifest.sample_id.astype(str)))}
    write_json(OUT/'stage_b_train_subset_lock.json',lock); return path,lock

def dataset_for_subset(manifest: pd.DataFrame) -> SteadMultitaskDataset:
    ds=SteadMultitaskDataset(INDEX,H5,split='train'); positions={str(v):i for i,v in enumerate(ds.metadata.trace_name)}
    if any(str(x) not in positions for x in manifest.sample_id): raise RuntimeError('subset sample absent from frozen train split')
    ds.metadata=ds.metadata.iloc[[positions[str(x)] for x in manifest.sample_id]].reset_index(drop=True); return ds
def full_validation_dataset() -> SteadMultitaskDataset: return SteadMultitaskDataset(INDEX,H5,split='val')
def loss_fn(out: dict[str,torch.Tensor], y: torch.Tensor) -> tuple[torch.Tensor,tuple[torch.Tensor,...]]:
    l=tuple(nn.functional.binary_cross_entropy_with_logits(out[k],y[:,i,:],reduction='mean') for i,k in enumerate(('detection_logits','p_logits','s_logits'))); return sum(l)/3,l
def phase_value(meta: dict[str,Any], key: str) -> int:
    v=meta.get(key,meta.get('trace_'+key,-1)); return int(v) if pd.notna(v) and 0<=float(v)<6000 else -1

def score(model: nn.Module, data: DataLoader, device: torch.device) -> tuple[pd.DataFrame,float]:
    model.eval(); rows=[]; losses=[]
    with torch.no_grad():
        for b in data:
            x,y=b['waveform'].to(device,non_blocking=True),b['labels'].to(device,non_blocking=True); o=model(x)
            if not all(torch.isfinite(v).all().item() for v in o.values()): raise RuntimeError('nonfinite validation output')
            total,_=loss_fn(o,y); losses.append(float(total.cpu()))
            probs={k:torch.sigmoid(v).cpu() for k,v in o.items()}
            for i,m in enumerate(b['metadata']):
                p,s=phase_value(m,'p_arrival_sample'),phase_value(m,'s_arrival_sample')
                rows.append({'is_event_label':int(y[i,0].max().item()>0),'detection_max_prob':float(probs['detection_logits'][i].max()),'p_max_prob':float(probs['p_logits'][i].max()),'s_max_prob':float(probs['s_logits'][i].max()),'p_pred_sample':int(probs['p_logits'][i].argmax()),'s_pred_sample':int(probs['s_logits'][i].argmax()),'p_true_sample':p,'s_true_sample':s,'p_true_available':int(p>=0),'s_true_available':int(s>=0)})
    return pd.DataFrame(rows),float(np.mean(losses))
def thresholds(df: pd.DataFrame) -> dict[str,float]:
    fixed={'detection':.5,'p':.5,'s':.5}; result={}
    for task in ('detection','p','s'):
        candidates=[]
        for t in GRID:
            m=metric_track(df,{**fixed,task:t})[task]; candidates.append((m['f1'],m['recall'],-t,t))
        result[task]=max(candidates)[3]
    return result
def summary_metrics(df: pd.DataFrame, th: dict[str,float], val_loss: float) -> dict[str,Any]:
    m=metric_track(df,th); m['validation_total_bce_loss']=val_loss; m['joint_f1']=(m['detection']['f1']+m['p']['f1']+m['s']['f1'])/3; m['ps_mean_f1']=(m['p']['f1']+m['s']['f1'])/2; return m
def best_key(m: dict[str,Any], epoch: int) -> tuple: return (m['joint_f1'],m['ps_mean_f1'],m['s']['recall'],-m['validation_total_bce_loss'],-epoch)

def protocol(subset_lock: dict[str,Any], stage_a: dict[str,Any]) -> dict[str,Any]:
    files=[ROOT/'scripts/public_benchmark/models/ab_lightweight_joint_picker.py',Path(__file__),ROOT/'scripts/public_benchmark/data/stead_multitask_dataset.py',ROOT/'scripts/public_benchmark/data/label_generator.py',ROOT/'scripts/public_benchmark/train_stead_formal_single_seed_v0.py']
    d={'candidates':list(ORDER),'model_code_sha256':sha_file(files[0]),'training_script_sha256':sha_file(files[1]),'train_subset_sha256':subset_lock['sha256'],'validation_split_order_sha256':sha_text('\n'.join(pd.read_csv(INDEX).query("split == 'val'").trace_name.astype(str))),'source_files':{str(x.relative_to(ROOT)):sha_file(x) for x in files[2:]},'input_preprocessing':'per-channel (x-mean)/max(std,1e-6), existing SteadMultitaskDataset','labels':'existing LabelConfig/make_multitask_labels: detection P-100:S+500, P/S Gaussian sigma 10 samples','evaluator':'metric_track from existing train_stead_formal_single_seed_v0.py; global argmax, <=50 samples; outside-tolerance is FP+FN','seed':SEED,'fp32':True,'cudnn_benchmark':False,'cudnn_deterministic':True,'tf32':False,'loss':'mean BCEWithLogits for D/P/S; total=(D+P+S)/3','optimizer':{'name':'AdamW','lr':1e-3,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,'amsgrad':False},'scheduler':{'name':'CosineAnnealingLR','T_max':15,'eta_min':1e-5,'step':'once after each epoch'},'epochs':15,'batch_size':32,'dataloader':{'num_workers':WORKERS,'pin_memory':True,'persistent_workers':True,'train_shuffle':True,'validation_shuffle':False,'drop_last':False},'gradient_clip_max_norm':1.0,'augmentation':{'enabled':False},'checkpoint_selection':'max Joint_F1; tie <=1e-6 then PS_mean_F1, S recall, lower validation BCE, earlier epoch','threshold_grid':GRID,'threshold_selection':'max validation F1, then recall, then smaller threshold','stage_a_cpu':{x['model']:x['cpu_single_thread_batch1'] for x in stage_a['models']},'test_dataset_constructed':False,'test_dataloader_constructed':False,'test_files_accessed':False}
    d['protocol_sha256']=sha_text(json.dumps(d,sort_keys=True,separators=(',',':'))); return d

def run_candidate(name: str, train: DataLoader, val: DataLoader, lock: dict[str,Any], stage_a: dict[str,Any], device: torch.device) -> dict[str,Any]:
    set_seed(); torch.cuda.reset_peak_memory_stats(device); model=MODELS[name]().to(device); opt=AdamW(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,amsgrad=False); sch=CosineAnnealingLR(opt,T_max=15,eta_min=1e-5)
    run=OUT/name; run.mkdir(); history=[]; best=None; start=time.time(); finite=True
    for epoch in range(1,16):
        model.train(); vals=[]; epoch_start=time.time()
        for batch_idx,b in enumerate(train):
            opt.zero_grad(set_to_none=True); out=model(b['waveform'].to(device,non_blocking=True)); total,_=loss_fn(out,b['labels'].to(device,non_blocking=True))
            if not torch.isfinite(total): raise RuntimeError(f'NaN/Inf {name} epoch={epoch} batch={batch_idx}')
            total.backward(); grad=clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(grad): raise RuntimeError(f'NaN/Inf gradient {name} epoch={epoch} batch={batch_idx}')
            opt.step(); vals.append(float(total.detach().cpu()))
        df,vl=score(model,val,device); th=thresholds(df); metrics=summary_metrics(df,th,vl); record={'epoch':epoch,'train_total_bce_loss':float(np.mean(vals)),'validation_total_bce_loss':vl,'joint_f1':metrics['joint_f1'],'ps_mean_f1':metrics['ps_mean_f1'],'s_recall':metrics['s']['recall'],'threshold_detection':th['detection'],'threshold_p':th['p'],'threshold_s':th['s'],'epoch_seconds':time.time()-epoch_start}; history.append(record)
        if best is None or best_key(metrics,epoch)>best_key(best['metrics'],best['epoch']): best={'epoch':epoch,'metrics':metrics,'thresholds':th,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
        sch.step()
    ck={'model_state_dict':best['state'],'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'best_epoch':best['epoch'],'validation_thresholds':best['thresholds'],'validation_metrics':best['metrics'],'candidate':name,'train_subset_sha256':lock['train_subset_sha256'],'protocol_sha256':lock['protocol_sha256']}; torch.save(ck,run/'best_checkpoint.pt')
    pd.DataFrame(history).to_csv(run/'training_history.csv',index=False,lineterminator='\n'); result={'candidate':name,'best_epoch':best['epoch'],'checkpoint_sha256':sha_file(run/'best_checkpoint.pt'),'normal_convergence':history[-1]['train_total_bce_loss']<history[0]['train_total_bce_loss'],'nan_inf_free':finite,'training_seconds':time.time()-start,'mean_epoch_seconds':float(np.mean([x['epoch_seconds'] for x in history])),'max_gpu_memory_mb':float(torch.cuda.max_memory_allocated(device)/1024**2),'validation':best['metrics'],'efficiency_stage_a':next(x for x in stage_a['models'] if x['model']==name),'protocol_sha256':lock['protocol_sha256']}; write_json(run/'metrics.json',result); (run/'run.log').write_text(json.dumps({'candidate':name,'completed_epochs':15,'best_epoch':best['epoch'],'status':'completed'})+'\n'); return result

def select(results: list[dict[str,Any]]) -> tuple[dict[str,Any],str,str]:
    by={r['candidate']:r for r in results}
    def ps(n): return by[n]['validation']['ps_mean_f1']
    def sr(n): return by[n]['validation']['s']['recall']
    def width(a,b): return b if ps(b)-ps(a)>=.0015 or sr(b)-sr(a)>=.0020 else a
    a,b=width('A-W1','A-W2'),width('B-W1','B-W2'); best_d=max(r['validation']['detection']['f1'] for r in results)
    def safe(n,o): return best_d-by[n]['validation']['detection']['f1']<=.003 or ps(n)-ps(o)>=.003
    sa,sb=safe(a,b),safe(b,a); decision={'A_selected':a,'B_selected':b,'detection_f1_best':best_d,'detection_safe':{a:sa,b:sb}}
    if sa and not sb: return decision|{'status':'PASS_STAGE_B_UNIQUE_CANDIDATE_SELECTED','selected_candidate':a},a,'A only passes detection safety gate'
    if sb and not sa: return decision|{'status':'PASS_STAGE_B_UNIQUE_CANDIDATE_SELECTED','selected_candidate':b},b,'B only passes detection safety gate'
    if not sa and not sb: return decision|{'status':'FAIL_STAGE_B_NO_ELIGIBLE_CANDIDATE','selected_candidate':None},'', 'neither selected width passes detection safety gate'
    pa,pb=by[a]['validation']['p']['f1'],by[b]['validation']['p']['f1']; sa_f,sb_f=by[a]['validation']['s']['f1'],by[b]['validation']['s']['f1']; ca,cb=by[a]['efficiency_stage_a']['cpu_single_thread_batch1']['p95_ms'],by[b]['efficiency_stage_a']['cpu_single_thread_batch1']['p95_ms']; delta=ps(a)-ps(b)
    if pa>pb and sa_f>sb_f and ca<cb: winner,reason=a,'A dominates';
    elif pb>pa and sb_f>sa_f and cb<ca: winner,reason=b,'B dominates'
    elif abs(delta)<.001: winner,reason=(a,'PS tie band: lower CPU p95') if ca<cb else (b,'PS tie band: lower CPU p95')
    elif delta<.001: winner,reason=b,'GRU latency compensation not met'
    else:
        keys=[(ps(a),ps(b),True),(by[a]['validation']['s']['recall'],by[b]['validation']['s']['recall'],True),(-ca,-cb,True),(-by[a]['efficiency_stage_a']['parameters'],-by[b]['efficiency_stage_a']['parameters'],True)]; winner=b; reason='all non-dominant tie fields equal'
        for left,right,_ in keys:
            if left!=right: winner=a if left>right else b; reason='non-dominant fixed-priority comparison'; break
    return decision|{'status':'PASS_STAGE_B_UNIQUE_CANDIDATE_SELECTED','selected_candidate':winner,'delta_ps':delta},winner,reason

def main() -> None:
    if OUT.exists(): raise RuntimeError('refuse overwrite Stage B output root')
    if not STAGE_A.exists(): raise RuntimeError('BLOCKED_STAGE_B_PREREQUISITE_MISSING: Stage A metrics absent')
    stage_a=json.loads(STAGE_A.read_text()); expected={'A-W1':46707,'A-W2':55363,'B-W1':53475,'B-W2':63931}
    if stage_a.get('status')!='PASS_STAGE_A_IMPLEMENTATION_LOCK' or {r['model']:r['parameters'] for r in stage_a['models']}!=expected: raise RuntimeError('BLOCKED_STAGE_B_PREREQUISITE_MISSING: Stage A mismatch')
    OUT.mkdir(parents=True); subset_path,subset_lock=build_subset(); manifest=pd.read_csv(subset_path); lock=protocol(subset_lock,stage_a); write_json(OUT/'stage_b_training_protocol_lock.json',lock)
    device=torch.device('cuda'); train=loader(dataset_for_subset(manifest),True); val=loader(full_validation_dataset(),False); results=[]
    try:
        for name in ORDER: results.append(run_candidate(name,train,val,lock,stage_a,device))
    except Exception as exc:
        write_json(OUT/'selected_final_candidate.json',{'status':'FAIL_STAGE_B_INCOMPLETE_RUN_SET','selected_candidate':None,'error':repr(exc)}); raise
    decision,winner,reason=select(results); pd.DataFrame([{'candidate':r['candidate'],'best_epoch':r['best_epoch'],'detection_f1':r['validation']['detection']['f1'],'p_f1':r['validation']['p']['f1'],'s_f1':r['validation']['s']['f1'],'s_recall':r['validation']['s']['recall'],'ps_mean_f1':r['validation']['ps_mean_f1'],'cpu_p95_ms':r['efficiency_stage_a']['cpu_single_thread_batch1']['p95_ms']} for r in results]).to_csv(OUT/'stage_b_all_results.csv',index=False,lineterminator='\n'); write_json(OUT/'stage_b_all_results.json',results)
    (OUT/'width_selection_decision.md').write_text('Width rule applied independently: W2 requires PS_mean_F1 +0.0015 or S recall +0.0020; otherwise W1.\n'); (OUT/'context_module_selection_decision.md').write_text(reason+'\n'); write_json(OUT/'selected_final_candidate.json',decision); (OUT/'stage_b_run_log.md').write_text(decision['status']+'\n')
if __name__=='__main__': main()
