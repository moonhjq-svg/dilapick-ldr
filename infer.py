"""CPU inference with the paper's frozen models and validation thresholds."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from scripts.public_benchmark.models.bw1_2x2_mechanism_ablation import BW1MechanismAblation
from scripts.public_benchmark.models.dilapick_ldr import DILaPickLDR
from scripts.public_benchmark.models.dilapick_recovery_controls import make_control

ROOT=Path(__file__).resolve().parent
def model_for(name):
    if name=='DilaPick': return BW1MechanismAblation(tcn_enabled=True,adapter_enabled=False)
    if name=='LDR': return DILaPickLDR()
    return make_control({'Linear':'LINEAR_CONTEXT','Narrow':'NARROW_FEATURE'}[name],3)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['DilaPick','LDR','Linear','Narrow'],default='LDR')
    p.add_argument('--seed',type=int,choices=[41,42,43],default=42)
    p.add_argument('--input',type=Path,help='Raw Z,N,E float array [3,6000], [6000,3], or [B,3,6000], 100 Hz')
    p.add_argument('--output',type=Path,default=Path('outputs/inference.npz'))
    a=p.parse_args(); torch.set_num_threads(1)
    cfg=json.loads((ROOT/f'configs/{a.model}_seed{a.seed}.json').read_text())
    model=model_for(a.model)
    raw=torch.load(ROOT/cfg['weights'],map_location='cpu',weights_only=True)
    state=raw.get('model_state_dict',raw.get('state_dict',raw))
    model.load_state_dict(state,strict=True); model.eval()
    x=np.load(a.input,allow_pickle=False) if a.input else np.random.default_rng(20261008).standard_normal((1,3,6000)).astype(np.float32)
    if x.shape==(6000,3): x=x.T
    if x.shape==(3,6000): x=x[None]
    if x.ndim!=3 or x.shape[1:]!=(3,6000) or not np.isfinite(x).all(): raise ValueError('Expected finite [B,3,6000] input')
    x=x.astype(np.float32); x=(x-x.mean(axis=-1,keepdims=True))/np.maximum(x.std(axis=-1,keepdims=True),1e-6)
    with torch.inference_mode():
        logits=model(torch.from_numpy(x))
        if isinstance(logits,dict): logits=torch.stack([logits[t+'_logits'] for t in ['detection','p','s']],dim=1)
        probs=logits.sigmoid().numpy()
    if probs.shape!=x.shape or not np.isfinite(probs).all(): raise RuntimeError('Invalid model output')
    peaks=probs.argmax(axis=-1); scores=probs.max(axis=-1)
    decisions=scores>=np.array([cfg['thresholds'][t] for t in ['detection','p','s']])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,probabilities=probs,peak_samples=peaks,peak_scores=scores,decisions=decisions)
    print(json.dumps(dict(status='PASS',model=a.model,seed=a.seed,synthetic_input=a.input is None,shape=list(probs.shape),parameters=sum(p.numel() for p in model.parameters()),thresholds=cfg['thresholds'],output=str(a.output))))
if __name__=='__main__': main()
