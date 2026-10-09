"""Check all distributed file hashes, including inference/scoring code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256SUMS.json').read_text())
for rel,expected in manifest.items():
    p=root/rel
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise SystemExit('FAIL: '+rel)
print(json.dumps({'status':'PASS','verified_files':len(manifest)}))
