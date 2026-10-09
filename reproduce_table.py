"""Recompute all 32 mean/SD cells of manuscript Table I from frozen predictions."""
from pathlib import Path
import csv, hashlib, json, re, statistics, argparse
ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('outputs')); a=p.parse_args()
    # Check the actual input bytes before scoring; this never selects a threshold.
    provenance=json.loads((ROOT/'evidence/provenance.json').read_text())
    for rec in provenance:
        f=ROOT/rec['path']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=rec['sha256']: raise ValueError('Hash mismatch: '+rec['path'])
    index=read(ROOT/'data/stead_50k_50k_index.csv'); test=[r for r in index if r['split']=='test']
    assert len(index)==len({r['trace_name'] for r in index})==100000
    assert len(test)==10000
    ids=[r['trace_name'] for r in test]; per_seed=[]; result=[]
    expected=(ROOT/'evidence/expected/table_recovery_controls.tex').read_text()
    for model in ['DilaPick','LDR','Linear','Narrow']:
        scores=[]
        for seed in [41,42,43]:
            cfg=json.loads((ROOT/f'configs/{model}_seed{seed}.json').read_text()); rows=read(ROOT/cfg['prediction'])
            assert [r['trace_name'] for r in rows]==ids and all(r['split']=='test' for r in rows)
            vals=[]
            for task in ['detection','p','s']:
                tp=fp=fn=0
                for r,meta in zip(rows,test):
                    truth=meta['sample_type']=='event'
                    assert bool(int(r['is_event_label']))==truth
                    pred=float(r[task+'_max_prob'])>=cfg['thresholds'][task]
                    if model=='LDR': assert pred==bool(int(r['valselected_'+task+'_decision']))
                    good=truth
                    if task!='detection':
                        assert bool(int(r[task+'_true_available']))==truth
                        good=truth and abs(int(r[task+'_pred_sample'])-int(round(float(meta['trace_'+task+'_arrival_sample']))))<=50
                    tp+=int(pred and good); fp+=int(pred and not good); fn+=int(truth and not (pred and good))
                f1=2*tp/(2*tp+fp+fn); vals.append(f1)
                per_seed.append(dict(model=model,seed=seed,task=task,tp=tp,fp=fp,fn=fn,f1=f1))
            vals.append((vals[1]+vals[2])/2); scores.append(vals)
        row={'model':model}; actual=[]
        for j,key in enumerate(['d_f1','p_f1','s_f1','ps_f1']):
            values=[v[j] for v in scores]
            row[key+'_mean']=statistics.mean(values); row[key+'_sd']=statistics.stdev(values)
            actual.extend([f'{row[key+"_mean"]:.6f}',f'{row[key+"_sd"]:.6f}'])
        line=next(x for x in expected.splitlines() if x.startswith(model+' &'))
        target=re.findall(r'\d+\.\d{6}',line)
        if actual!=target: raise ValueError(f'{model}: computed {actual} != manuscript {target}')
        result.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,records in [('table_recovery_controls.csv',result),('per_seed_metrics.csv',per_seed)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    report=dict(status='PASS',table='Table I: recovery strategies',matched_mean_sd_cells=32,models=4,seeds=[41,42,43],predictions_per_model_seed=10000,reference_arrivals='round original metadata; 100 Hz; tolerance 50 samples',other_tables='not recomputed by this minimal release',training_rerun=False)
    (a.output/'table_verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__': main()
