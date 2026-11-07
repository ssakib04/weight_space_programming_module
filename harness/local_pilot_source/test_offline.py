from pathlib import Path
import ast, tempfile, json, hashlib
import nbformat
D=Path(__file__).resolve().parent; root=D.parent
for p in D.glob('*.py'): ast.parse(p.read_text())
nb=nbformat.read(D/'wsp_local_strong_pilot.ipynb',as_version=4);nbformat.validate(nb)
for c in nb.cells:
 if c.cell_type=='code':compile(c.source,'<notebook-cell>','exec')
assert all('import anthropic' not in c.source.lower() and 'anthropic.Anthropic' not in c.source and 'kaggle_secrets' not in c.source and 'requests.post' not in c.source for c in nb.cells)
ns={'__name__':'offline_test'}
exec((D/'config.py').read_text(),ns)
for f in ['10_tasks.py','20_harness.py']:exec((root/'v3_src'/f).read_text(),ns)
assert len(ns['PILOT_CASES'])==6 and all(level=='none' for task,level,rep in ns['PILOT_CASES'])
fixture=json.loads((D/'exact_no_hint_prompts.json').read_text())
for task in ['modmul','permadd']:
 assert fixture['messages'][task][1]['content']==ns['task_prompt'](task,'none')
exec((root/'v3_src/50_B.py').read_text().split('def synth_once(')[0],ns)
results={}
for task in ['modmul','permadd']:
 src='```python\nimport numpy as np\n'+ns['ORACLE_SRC'][task]+'\n```'
 row=ns['process_reply'](task,src,{'stop_reason':'end_turn'},'offline_'+task)
 assert row['ok'] and row['success'],row
 bad=ns['process_reply'](task,'no code',{'stop_reason':'end_turn'},'offline_bad_'+task)
 assert not bad['ok'] and not bad['success']
 trunc=ns['process_reply'](task,'def build_weights():\n  x = (',{'stop_reason':'max_tokens'},'offline_trunc_'+task)
 assert trunc['error_kind']=='truncated',trunc
 results[task]={'oracle_accuracy':row['all_acc'],'no_code':bad['error_kind'],'truncation':trunc['error_kind']}
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'v3_src/10_tasks.py',root/'v3_src/20_harness.py',root/'v3_src/50_B.py',D/'config.py',D/'runner.py',D/'wsp_local_strong_pilot.ipynb',D/'exact_no_hint_prompts.json']}
(D/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
(D/'OFFLINE_TEST_RESULTS.json').write_text(json.dumps({'checks':'syntax, notebook schema, exact no-hint prompt fixture/config consistency, no paid provider code, CPU task oracles, valid replies, missing code, truncated code','results':results,'not_tested':'real model loading, pinned Kaggle dependency installation, T4 VRAM, download disk, generation (exact tokenizers tested separately under 4.51.3)'},indent=2))
print('OFFLINE PASS',results)
