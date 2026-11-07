"""Offline verification of the release candidate. No network, no API calls, no training, no model-generated code is executed.
Run from anywhere:  python scripts/verify_release.py   [--skip-figures]"""
import hashlib, json, re, sys, shutil, subprocess, tempfile
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness')); sys.path.insert(0, str(ROOT / 'scripts'))
from wsp_eval import evaluate_npz
import scan_secrets
RESULTS = []
def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail)); print(('PASS ' if cond else 'FAIL ') + name + (f' — {detail}' if detail else ''))

# 1. File integrity
man = ROOT / 'MANIFEST.sha256'; bad = []; n = 0
for line in man.read_text().splitlines():
    h, rel = line.split('  ', 1); n += 1
    if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h: bad.append(rel)
listed = {l.split('  ', 1)[1] for l in man.read_text().splitlines()}
extra = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and str(p.relative_to(ROOT)) not in listed | {'MANIFEST.sha256'}
         and '__pycache__' not in p.parts and not str(p.relative_to(ROOT)).startswith('verification_output/')]
check('manifest sha256', not bad and not extra, f'{n} files; mismatched={bad[:3]} unlisted={extra[:3]}')

# 2. Frozen Cursor prompts byte-stable
pm = json.loads((ROOT / 'prompts/cursor_frozen/prompts_start_here_manifest.json').read_text())['files']
ok = all(hashlib.sha256((ROOT / 'prompts/cursor_frozen' / f).read_bytes()).hexdigest() == v['sha256'] for f, v in pm.items())
check('frozen Cursor prompt hashes', ok, f'{len(pm)} files')

# 3. Cursor: all 30 saved weight files rescored offline
trials = json.loads((ROOT / 'cursor_trials/ALL30_per_trial.json').read_text())['trials']
rows = []; mism = []
for t in trials:
    d = ROOT / 'cursor_trials' / t['trial_id']
    f = d / '05_weights.npz' if (d / '05_weights.npz').exists() else d / '07_weights.npz'
    rep = json.loads((d / t['report_file']).read_text())
    r32 = evaluate_npz(f, t['task'], np.float32); r64 = evaluate_npz(f, t['task'], np.float64)
    rec64 = (rep.get('np64') or rep.get('float64_numpy_eval') or {}).get('correct')
    rec_succ = rep['success'] if 'success' in rep else (rep.get('as_is_contract_outcome') or rep.get('diagnostic_transparent_extraction'))['success']
    if r32['correct'] != t['correct'] or r32['success'] != t['success'] or rec_succ != t['success'] or (rec64 is not None and rec64 != r64['correct']):
        mism.append(t['trial_id'])
    rows.append({'trial_id': t['trial_id'], 'model_label_reported': t['model_label_reported'], 'task': t['task'], 'weights_file': str(f.relative_to(ROOT)),
                 'reported_correct': t['correct'], 'rescored_correct_float32': r32['correct'], 'rescored_correct_float64': r64['correct'],
                 'total': r32['total'], 'accuracy_float32': r32['accuracy'], 'success': r32['success']})
df = pd.DataFrame(rows)
out = ROOT / 'verification_output'; out.mkdir(exist_ok=True); df.to_csv(out / 'cursor30_offline_rescore.csv', index=False)
cells = df.assign(m=df.trial_id.str.split('_').str[0]).groupby(['m', 'task']).success.sum().to_dict()
exp = {('sonnet55high','modmul'):4, ('sonnet55high','permadd'):5, ('gemini4argonmedium','modmul'):3, ('gemini4argonmedium','permadd'):5, ('gpt56solmedium','modmul'):5, ('gpt56solmedium','permadd'):5}
check('Cursor 30 weights rescored = reports', len(df) == 30 and not mism, f'mismatches={mism}')
check('Cursor primary endpoint 27/30 and per-cell counts', int(df.success.sum()) == 27 and cells == exp, str(cells))
check('Cursor failures are the 3 documented trials', sorted(df[~df.success].trial_id) == ['gemini4argonmedium_modmul_none_r2', 'gemini4argonmedium_modmul_none_r4', 'sonnet55high_modmul_none_r3'])
check('attestation addendum covers all 30, not marked verified', (lambda a: len(a['trials']) == 30 and 'NOT independently verified' in a['verification_status'])(json.loads((ROOT / 'cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json').read_text())))

# 4. API Claude (separate channel; never pooled with Cursor/open models)
api = pd.read_csv(ROOT / 'results/tables/final_claude_per_run.csv'); alg = api[api.task.isin(['modadd', 'modmul', 'parity'])]
mm = alg[alg.task == 'modmul'].groupby('prompt').success.sum().to_dict()
check('API algorithmic 92/105', len(alg) == 105 and int(alg.success.sum()) == 92)
check('API modmul 38/45 (full 12, partial 13, none 13)', int(alg[alg.task == 'modmul'].success.sum()) == 38 and mm == {'full': 12, 'none': 13, 'partial': 13}, str(mm))
r2 = pd.read_csv(ROOT / 'runs/v3_run2_outputs/per_run.csv'); n2 = bad2 = 0
for f in sorted((ROOT / 'runs/v3_run2_outputs/generated_code').glob('*_modmul_*_weights.npz')):
    task, pr, rep = re.search(r'_(modmul)_(full|partial|none)_r(\d+)_weights', f.name).groups()
    row = r2[(r2.instance == 'B') & (r2.task == task) & (r2.prompt == pr) & (r2.rep == int(rep))].iloc[0]
    r = evaluate_npz(f, task); n2 += 1; bad2 += abs(r['accuracy'] - row.all_acc) > 1e-9 or r['success'] != bool(row.success)
check('API Run-2 fresh modmul saved weights rescored = logged', n2 == 15 and bad2 == 0, f'{n2} files')

# 5. Open-weight models (separate)
lp = pd.read_csv(ROOT / 'runs/local_pilot_phi4_mistral/per_run.csv'); lb = lp[lp.instance == 'B']
check('Phi-4/Mistral pilot 0/12; 5 runnable, 7 failed', len(lb) == 12 and int(lb.success.sum()) == 0 and int(lb.ok.sum()) == 5)
nl = badl = 0
for f in sorted((ROOT / 'runs/local_pilot_phi4_mistral/generated_code').glob('*_weights.npz')):
    task, pr, rep = re.search(r'_(modmul|permadd)_(none)_r(\d+)_weights', f.name).groups(); syn = 'microsoft/phi-4' if 'phi-4' in f.name else 'mistralai/Mistral-Small-24B-Instruct-2501'
    row = lb[(lb.synth == syn) & (lb.task == task) & (lb.rep == int(rep))].iloc[0]; r = evaluate_npz(f, task); nl += 1
    badl += abs(r['accuracy'] - row.all_acc) > 1e-9 or r['success']
check('Phi/Mistral saved weights rescored = logged (all fail)', nl == 5 and badl == 0, f'{nl} files')
r1 = pd.read_csv(ROOT / 'runs/v3_run1_outputs/per_run.csv'); q = r1[(r1.instance == 'B') & r1.synth.str.startswith('Qwen/') & r1.task.isin(['modadd', 'modmul', 'parity', 'permadd'])]
check('Qwen Run-1 algorithmic 0/72 (2048-token cap, reported separately)', len(q) == 72 and int(q.success.sum()) == 0)
nq = badq = 0
for f in sorted((ROOT / 'runs/v3_run1_outputs/generated_code').glob('Qwen_*_weights.npz')):
    m = re.search(r'Qwen_(Qwen2\.5-7B-Instruct|Qwen2\.5-Coder-14B-Instruct)_(modadd|modmul|permadd)_(full|partial|none)_r(\d+)_weights', f.name)
    if not m: continue
    syn, task, pr, rep = m.groups(); row = q[(q.synth == 'Qwen/' + syn) & (q.task == task) & (q.prompt == pr) & (q.rep == int(rep))]
    if len(row) != 1: badq += 1; continue
    r = evaluate_npz(f, task); nq += 1
    if r.get('ok'): badq += abs(r['accuracy'] - row.iloc[0].all_acc) > 1e-9 or r['success']
check('Qwen saved weights rescored = logged (all fail)', nq > 0 and badq == 0, f'{nq} files')

# 6. Secrets / private details
h = scan_secrets.scan(ROOT)
check('secret and private-detail scan clean', not h, ', '.join(f'{k}:{sum(n for _, n in v)}' for k, v in h.items()))

# 7. Figures rebuild from packaged data (optional; needs matplotlib)
if '--skip-figures' not in sys.argv:
    try:
        import matplotlib  # noqa
        tmp = Path(tempfile.mkdtemp()) / 'paper'; shutil.copytree(ROOT / 'paper', tmp)
        shutil.rmtree(tmp / 'figures'); (tmp / 'figures').mkdir()
        res = subprocess.run([sys.executable, str(tmp / 'scripts/build_figures.py')], capture_output=True, text=True, cwd=tmp)
        made = sorted(p.stem for p in (tmp / 'figures').glob('*.png'))
        check('figures rebuild from packaged data', res.returncode == 0 and len(made) == 10, f'{len(made)} PNGs' + ('' if res.returncode == 0 else ' ' + res.stderr[-300:]))
    except ImportError:
        print('SKIP figures rebuild (matplotlib not installed)')
fails = [r for r in RESULTS if not r[1]]
(out / 'verification_summary.json').write_text(json.dumps([{'check': a, 'pass': b, 'detail': c} for a, b, c in RESULTS], indent=1))
print(f'\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed'); sys.exit(1 if fails else 0)
