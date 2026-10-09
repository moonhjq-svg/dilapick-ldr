"""BW1_PROSPECTIVE_FULL_PROTOCOL_V1 2x2 TCN/adapter ablation, seed 41 only."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path('/home/u703/Hjq')
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.public_benchmark.models.ab_lightweight_joint_picker import BW1
from scripts.public_benchmark.models.bw1_2x2_mechanism_ablation import BW1MechanismAblation
from scripts.public_benchmark.data.stead_multitask_dataset import SteadMultitaskDataset
from scripts.public_benchmark.train_ab_lightweight_joint_picker_stage_b import H5, INDEX, GRID, loader, loss_fn, score, set_seed, summary_metrics, thresholds, sha_file, sha_text, write_json

OUT = ROOT/'experiments/public_benchmark_restart/bw1_2x2_mechanism_ablation_seed41'
FULL = ROOT/'experiments/public_benchmark_restart/bw1_full_seed41'
LOCK = FULL/'full_training_protocol_lock.json'
CKPT = FULL/'best_checkpoint.pt'
MODEL_SOURCE = ROOT/'scripts/public_benchmark/models/ab_lightweight_joint_picker.py'
ABLATION_SOURCE = ROOT/'scripts/public_benchmark/models/bw1_2x2_mechanism_ablation.py'
TRAINER_SOURCE = Path(__file__)
CONFIGS = [('M00', False, False), ('M10', True, False), ('M01', False, True), ('M11', True, True)]
RESULT_COLUMNS = ['model','tcn_enabled','adapter_enabled','best_epoch','detection_precision','detection_recall','detection_f1','p_precision','p_recall','p_f1','p_mae','p_missed','s_precision','s_recall','s_f1','s_mae','s_missed','ps_mean_f1','parameters','mac','fp32_mib','cpu_p50_ms','cpu_p95_ms','checkpoint_sha256']

def jwrite(path: Path, obj: dict) -> None: write_json(path, obj)
def file_sha(p: Path) -> str: return sha_file(p)
def count(model): return int(sum(p.numel() for p in model.parameters()))
def macs(model):
    total = 0; hooks=[]
    def f(m, _i, o):
        nonlocal total
        if isinstance(m, torch.nn.Conv1d):
            n,c,l=o.shape; total += int(n*c*l*(m.in_channels//m.groups)*m.kernel_size[0])
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d): hooks.append(m.register_forward_hook(f))
    with torch.no_grad(): model.eval()(torch.zeros(1,3,6000))
    for h in hooks: h.remove()
    return total
def fp32_mib(model): return float(sum(p.numel()*p.element_size() for p in model.parameters())/1024**2)
def rank(m, epoch): return (m['joint_f1'],m['ps_mean_f1'],m['s']['recall'],-m['validation_total_bce_loss'],-epoch)
def row_from_metric(name,tcn,adapter,metric,best_epoch,ck_sha,eff):
    return {'model':name,'tcn_enabled':tcn,'adapter_enabled':adapter,'best_epoch':best_epoch,
      'detection_precision':metric['detection']['precision'],'detection_recall':metric['detection']['recall'],'detection_f1':metric['detection']['f1'],
      'p_precision':metric['p']['precision'],'p_recall':metric['p']['recall'],'p_f1':metric['p']['f1'],'p_mae':metric['p']['mae_seconds'],'p_missed':metric['p']['false_negative_count'],
      's_precision':metric['s']['precision'],'s_recall':metric['s']['recall'],'s_f1':metric['s']['f1'],'s_mae':metric['s']['mae_seconds'],'s_missed':metric['s']['false_negative_count'],
      'ps_mean_f1':metric['ps_mean_f1'],'parameters':eff['parameters'],'mac':eff['mac'],'fp32_mib':eff['fp32_mib'],'cpu_p50_ms':eff['cpu_p50_ms'],'cpu_p95_ms':eff['cpu_p95_ms'],'checkpoint_sha256':ck_sha}

def cpu_worker(tcn: bool, adapter: bool):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    m=BW1MechanismAblation(tcn_enabled=tcn, adapter_enabled=adapter).cpu().eval(); x=torch.randn(1,3,6000)
    try:
        import psutil; proc=psutil.Process(os.getpid()); peak=proc.memory_info().rss
    except Exception: proc=None; peak=None
    with torch.no_grad():
        for _ in range(100): m(x)
        v=[]
        for _ in range(500):
            t=time.perf_counter_ns(); m(x); v.append((time.perf_counter_ns()-t)/1e6)
            if proc is not None: peak=max(peak,proc.memory_info().rss)
    print(json.dumps({'p50_ms':float(np.percentile(v,50)),'p95_ms':float(np.percentile(v,95)),'peak_rss_mb':None if peak is None else peak/1024**2}))

def cpu_audit(tcn, adapter):
    vals=[]
    for _ in range(3):
        r=subprocess.run([sys.executable, str(TRAINER_SOURCE), '--cpu-worker', str(int(tcn)), str(int(adapter))],capture_output=True,text=True,check=True)
        vals.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {'cpu_p50_ms':float(np.median([x['p50_ms'] for x in vals])),'cpu_p95_ms':float(np.median([x['p95_ms'] for x in vals])),'peak_rss_mb':next((x['peak_rss_mb'] for x in vals if x['peak_rss_mb'] is not None),None),'cpu_repeats':vals}

def m11_equivalence():
    raw=torch.load(CKPT,map_location='cpu',weights_only=False); state=raw['model_state_dict']
    old=BW1().eval(); new=BW1MechanismAblation(tcn_enabled=True,adapter_enabled=True).eval()
    old.load_state_dict(state,strict=True); new.load_state_dict(state,strict=True)
    torch.manual_seed(41); results={}; ok=count(new)==53475 and macs(new)==72216000
    for b in (1,2):
        x=torch.randn(b,3,6000)
        with torch.no_grad(): a=old(x); z=new(x)
        errs={k:float((a[k]-z[k]).abs().max()) for k in a}; results[str(b)]={'shapes':{k:list(v.shape) for k,v in z.items()},'max_abs_error':errs}; ok=ok and all(e<=1e-7 for e in errs.values()) and all(tuple(v.shape)==(b,6000) for v in z.values())
    return {'status':'PASS' if ok else 'BLOCKED_BW1_2X2_M11_EQUIVALENCE_FAILED','original_model_code_sha256':file_sha(MODEL_SOURCE),'new_2x2_model_code_sha256':file_sha(ABLATION_SOURCE),'checkpoint_sha256':file_sha(CKPT),'strict_state_dict_load':True,'parameters':count(new),'mac':macs(new),'batches':results}

def protocol_lock(eq):
    x=json.loads(LOCK.read_text())
    return {'protocol_name':x['protocol_name'],'parent_protocol_sha256':x['protocol_sha256'],'seed':41,'train_split_sha256':x['split_sha256']['train'],'validation_split_sha256':x['split_sha256']['validation'],'fp32':True,'tf32':False,'amp':False,'epochs':20,'optimizer':x['optimizer'],'scheduler':x['scheduler'],'batch_size':x['batch_size'],'gradient_clip_max_norm':x['gradient_clip_max_norm'],'dataloader':x['dataloader'],'augmentation':x['augmentation'],'loss':x['loss'],'checkpoint_selection':x['checkpoint_selection'],'threshold_grid':x['threshold_grid'],'threshold_selection':x['threshold_selection'],'test_dataset_constructed':False,'test_dataloader_constructed':False,'test_files_accessed':False,'fixed_configs':{n:{'tcn_enabled':t,'adapter_enabled':a} for n,t,a in CONFIGS},'m11_equivalence_sha256':sha_text(json.dumps(eq,sort_keys=True,separators=(',',':')))}

def train_one(name,tcn,adapter,lock):
    d=OUT/name; d.mkdir(); set_seed(); device=torch.device('cuda'); train=loader(SteadMultitaskDataset(INDEX,H5,split='train'),True); val=loader(SteadMultitaskDataset(INDEX,H5,split='val'),False)
    model=BW1MechanismAblation(tcn_enabled=tcn,adapter_enabled=adapter).to(device); opt=AdamW(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,amsgrad=False); sch=CosineAnnealingLR(opt,T_max=20,eta_min=1e-5); torch.cuda.reset_peak_memory_stats(device); hist=[]; best=None; started=time.time()
    try:
        for epoch in range(1,21):
            tick=time.time(); model.train(); losses=[]
            for bi,b in enumerate(train):
                opt.zero_grad(set_to_none=True); o=model(b['waveform'].to(device,non_blocking=True)); total,_=loss_fn(o,b['labels'].to(device,non_blocking=True))
                if not torch.isfinite(total): raise RuntimeError(f'NaN/Inf loss epoch={epoch} batch={bi}')
                total.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(norm): raise RuntimeError(f'NaN/Inf gradient epoch={epoch} batch={bi}')
                opt.step(); losses.append(float(total.detach().cpu()))
            df,vl=score(model,val,device); th=thresholds(df); metric=summary_metrics(df,th,vl); hist.append({'epoch':epoch,'train_total_bce_loss':float(np.mean(losses)),'validation_total_bce_loss':vl,'joint_f1':metric['joint_f1'],'ps_mean_f1':metric['ps_mean_f1'],'s_recall':metric['s']['recall'],'threshold_detection':th['detection'],'threshold_p':th['p'],'threshold_s':th['s'],'epoch_seconds':time.time()-tick})
            if best is None or rank(metric,epoch)>rank(best['metrics'],best['epoch']): best={'epoch':epoch,'metrics':metric,'thresholds':th,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            sch.step()
    except Exception as e:
        jwrite(d/'metrics.json',{'status':'FAIL_BW1_2X2_INCOMPLETE_RUN_SET','model':name,'error':repr(e),'test_dataset_constructed':False,'test_dataloader_constructed':False}); (d/'run.log').write_text(repr(e)+'\n'); raise
    ck={'model_state_dict':best['state'],'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'best_epoch':best['epoch'],'validation_thresholds':best['thresholds'],'validation_metrics':best['metrics'],'model':name,'protocol_sha256':lock['parent_protocol_sha256']}; torch.save(ck,d/'best_checkpoint.pt'); pd.DataFrame(hist).to_csv(d/'training_history.csv',index=False,lineterminator='\n')
    result={'status':'COMPLETE','model':name,'tcn_enabled':tcn,'adapter_enabled':adapter,'best_epoch':best['epoch'],'checkpoint_sha256':file_sha(d/'best_checkpoint.pt'),'normal_convergence':hist[-1]['train_total_bce_loss']<hist[0]['train_total_bce_loss'],'nan_inf_free':True,'training_seconds':time.time()-started,'max_gpu_memory_mb':float(torch.cuda.max_memory_allocated(device)/1024**2),'validation':best['metrics'],'protocol_sha256':lock['parent_protocol_sha256'],'test_dataset_constructed':False,'test_dataloader_constructed':False,'test_files_accessed':False}; jwrite(d/'metrics.json',result); (d/'run.log').write_text('COMPLETE\n'); return result

def finalise(lock,eq,results):
    efficiencies={};
    for n,t,a in CONFIGS: 
        m=BW1MechanismAblation(tcn_enabled=t,adapter_enabled=a); c=cpu_audit(t,a); efficiencies[n]={'parameters':count(m),'mac':macs(m),'fp32_mib':fp32_mib(m),**c}
    full=json.loads((FULL/'metrics.json').read_text()); results['M11']={'best_epoch':full['best_epoch'],'checkpoint_sha256':file_sha(CKPT),'normal_convergence':full['normal_convergence'],'nan_inf_free':full['nan_inf_free'],'training_seconds':full['training_seconds'],'max_gpu_memory_mb':full['max_gpu_memory_mb'],'validation':full['validation']}
    rows=[]
    for n,t,a in CONFIGS: rows.append(row_from_metric(n,t,a,results[n]['validation'],results[n]['best_epoch'],results[n]['checkpoint_sha256'],efficiencies[n]))
    with open(OUT/'four_model_results.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=RESULT_COLUMNS);w.writeheader();w.writerows(rows)
    jwrite(OUT/'four_model_results.json',{'protocol':lock,'m11_equivalence':eq,'results':rows,'per_model':results,'efficiency_detail':efficiencies})
    r={x['model']:x for x in rows}; metrics=['detection_f1','p_f1','s_f1','p_recall','s_recall','ps_mean_f1']; main=[]; interaction=[]
    for k in metrics:
        main.extend([{'metric':k,'effect':'Effect_TCN','value':r['M10'][k]-r['M00'][k]},{'metric':k,'effect':'Effect_Adapter','value':r['M01'][k]-r['M00'][k]},{'metric':k,'effect':'Effect_TCN_given_adapter','value':r['M11'][k]-r['M01'][k]},{'metric':k,'effect':'Effect_Adapter_given_TCN','value':r['M11'][k]-r['M10'][k]}]); interaction.append({'metric':k,'interaction':r['M11'][k]-r['M10'][k]-r['M01'][k]+r['M00'][k]})
    pd.DataFrame(main).to_csv(OUT/'mechanism_main_effects.csv',index=False); pd.DataFrame(interaction).to_csv(OUT/'mechanism_interaction_effects.csv',index=False)
    eff=[]
    for n in ('M10','M01','M11'):
        dp=r[n]['parameters']-r['M00']['parameters']; dt=r[n]['cpu_p95_ms']-r['M00']['cpu_p95_ms']; gain=r[n]['ps_mean_f1']-r['M00']['ps_mean_f1']; eff.append({'model':n,'ps_gain_vs_m00':gain,'added_parameters':dp,'added_cpu_p95_ms':dt,'ps_gain_per_added_Mparam':'N/A' if dp==0 else gain/(dp/1e6),'ps_gain_per_added_ms':'N/A' if dt==0 else gain/dt,'ps_per_million_parameters':r[n]['ps_mean_f1']/(r[n]['parameters']/1e6),'ps_per_cpu_p95_ms':r[n]['ps_mean_f1']/r[n]['cpu_p95_ms']})
    pd.DataFrame(eff).to_csv(OUT/'efficiency_increment_report.csv',index=False)
    i={x['metric']:x['interaction'] for x in interaction}; dom=lambda x: x['p_f1']>r['M11']['p_f1'] and x['s_f1']>r['M11']['s_f1'] and x['cpu_p95_ms']<r['M11']['cpu_p95_ms']
    strong=(r['M11']['ps_mean_f1']>r['M10']['ps_mean_f1'] and r['M11']['ps_mean_f1']>r['M01']['ps_mean_f1'] and r['M10']['ps_mean_f1']>r['M00']['ps_mean_f1'] and r['M01']['ps_mean_f1']>r['M00']['ps_mean_f1'] and i['ps_mean_f1']>=.001 and r['M11']['ps_mean_f1']-r['M00']['ps_mean_f1']>=.003 and r['M00']['detection_f1']-r['M11']['detection_f1']<=.001 and r['M11']['cpu_p95_ms']<=10 and (r['M11']['p_f1']==max(x['p_f1'] for x in rows) or r['M11']['s_f1']==max(x['s_f1'] for x in rows)) and not any(dom(x) for x in rows if x['model']!='M11'))
    complementary=(r['M11']['ps_mean_f1']==max(x['ps_mean_f1'] for x in rows) and r['M10']['ps_mean_f1']>r['M00']['ps_mean_f1'] and r['M01']['ps_mean_f1']>r['M00']['ps_mean_f1'] and r['M11']['ps_mean_f1']-r['M00']['ps_mean_f1']>=.002 and r['M00']['detection_f1']-r['M11']['detection_f1']<=.001 and i['ps_mean_f1']<.001 and r['M11']['cpu_p95_ms']<=10)
    insufficient=(r['M11']['ps_mean_f1']<=r['M10']['ps_mean_f1'] or r['M11']['ps_mean_f1']<=r['M01']['ps_mean_f1'] or (r['M10']['ps_mean_f1']<=r['M00']['ps_mean_f1'] and r['M01']['ps_mean_f1']<=r['M00']['ps_mean_f1']) or r['M11']['ps_mean_f1']-r['M00']['ps_mean_f1']<.001 or any(dom(x) for x in rows if x['model']!='M11'))
    gate='PASS_BW1_STRONG_SYNERGY_EVIDENCE' if strong else 'PASS_BW1_COMPLEMENTARY_COMPONENT_EVIDENCE' if complementary else 'FAIL_BW1_MECHANISM_STORY_INSUFFICIENT' if insufficient else 'FAIL_BW1_MECHANISM_STORY_INCONCLUSIVE'
    (OUT/'mechanism_evidence_decision.md').write_text(f'# {gate}\n\nPrimary preregistered endpoint: Interaction_PS_mean_F1 = {i["ps_mean_f1"]:.9f}.\nNo test was constructed, read, or run.\n')
    (OUT/'run.log').write_text(gate+'\n')

def main():
    if OUT.exists(): raise RuntimeError('output root already exists; no rerun permitted')
    OUT.mkdir(parents=True); eq=m11_equivalence(); jwrite(OUT/'m11_equivalence_check.json',eq)
    if eq['status']!='PASS': jwrite(OUT/'ablation_protocol_lock.json',{'status':eq['status']}); (OUT/'run.log').write_text(eq['status']+'\n'); return
    lock=protocol_lock(eq); jwrite(OUT/'ablation_protocol_lock.json',lock); (OUT/'M11_reference').mkdir(); jwrite(OUT/'M11_reference/frozen_checkpoint_reference.json',{'checkpoint_path':str(CKPT),'checkpoint_sha256':file_sha(CKPT)}); full=json.loads((FULL/'metrics.json').read_text()); jwrite(OUT/'M11_reference/frozen_metrics_reference.json',{'metrics_path':str(FULL/'metrics.json'),'best_epoch':full['best_epoch'],'validation':full['validation'],'checkpoint_sha256':file_sha(CKPT)}); jwrite(OUT/'M11_reference/equivalence_check.json',eq)
    results={}
    try:
        for n,t,a in CONFIGS[:3]: results[n]=train_one(n,t,a,lock)
    except Exception: (OUT/'run.log').write_text('FAIL_BW1_2X2_INCOMPLETE_RUN_SET\n'); raise
    finalise(lock,eq,results)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cpu-worker',nargs=2); z=ap.parse_args()
    if z.cpu_worker: cpu_worker(bool(int(z.cpu_worker[0])),bool(int(z.cpu_worker[1])))
    else: main()
