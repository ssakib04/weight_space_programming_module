from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify
import json,hashlib,pandas as pd
p=Path(__file__).resolve().parents[1];r=p.parent
f=p/'WSP_manuscript_v2.html';s=BeautifulSoup(f.read_text(),'html.parser')
for para in s.find_all('p'):
 text=para.get_text()
 if 'On a Cursor chat channel' in text and 'formatting exception' not in text:
  para.append(' The Cursor count is mathematical construction success after documented extraction, including one formatting exception, not a strict raw-response compliance rate.')
 if 'without training data or gradients' in text:
  for node in para.find_all(string=True):
   if 'accuracy on every input' in node:node.replace_with(str(node).replace('accuracy on every input','accuracy across the complete input domain'))
if not s.find(id='audit-extraction'):
 q=s.new_tag('p',id='audit-extraction');q.string=('Extraction policy. Cursor 27/30 counts mathematical construction success, not raw-response format compliance. In sonnet55high_modmul_none_r0, the strict Python-only path fails on a shell heredoc wrapper/self-test (NameError: EOF); the saved diagnostic extracts the unchanged build_weights function and achieves 12,544/12,544. This formatting exception is included in 27/30, without numerical repair, and must not be described as every raw response executing unchanged. The per-cell counts and Figure 2 use this construction endpoint. Provider identities and fresh-chat hygiene are not independently verified for Cursor.')
 s.find_all('table')[1].insert_after(q)
f.write_text(str(s))
(p/'WSP_manuscript_v2.md').write_text('---\nstatus: REVIEW-READY DRAFT\nauthor: Kazi Sadman Sakib\n---\n\n'+markdownify(str(s.body),heading_style='ATX',code_language='python'))
# Source-level assertions. Resolve the three nested r0 report schemas explicitly.
ts=json.loads((r/'cursor_trials/ALL30_per_trial.json').read_text())['trials']
for t in ts:
 d=json.loads((Path(t['dir'])/t['report_file']).read_text());x=d
 if 'success' not in x:x=d.get('as_is_contract_outcome',d.get('diagnostic_transparent_extraction'))
 assert x['success']==t['success'],t['trial_id']
assert len(ts)==30 and sum(x['success'] for x in ts)==27
api=pd.read_csv(r/'final/tables/final_claude_per_run.csv');api=api[api.task.isin(['modadd','modmul','parity'])];assert len(api)==105 and api.success.sum()==92
assert api[api.task=='modmul'].success.sum()==38
for pre,slug in zip(s.find_all('pre'),['gpt56solmedium_permadd_none_r0','sonnet55high_modmul_none_r3']):assert pre.get_text().strip()==(r/'cursor_trials'/slug/'00_raw_pasted_function_exact.py').read_text().strip()
assert len(s.find_all('figure'))==10 and len(s.find_all('table'))==3
for img in s.find_all('img'):assert (p/img['src']).exists()
a=pd.read_csv(p/'data/local_pilot_per_run.csv');a=a[a.instance=='B'];assert len(a)==12 and a.ok.sum()==5 and a.success.sum()==0
assert (p/'data/A_curves_run1.csv').read_bytes()==(r/'v3_run1_outputs/A_curves.csv').read_bytes()
# Detailed manifest addendum, with hashes so copied data can be checked without trust.
manifest=p/'SOURCE_TO_FIGURE_MANIFEST.md'
text=manifest.read_text().split('## Independent audit details')[0]
text+='\n## Independent audit details\n\n'
text+='Figure 1: exact byte-identical copy of v3_run1_outputs/A_curves.csv, 468 rows. LR chosen independently per task/optimizer by maximum mean final validation accuracy across seeds 0–2; curves are checkpoint validation means ± sample SD (not CI). No test-based selection. Algorithmic split is 30/10/60; CIFAR is 45k/5k plus official 10k test. Lines join checkpoints only.\n\n'
text+='Figure 2: construction endpoint includes the documented Sonnet modmul r0 formatting-only extraction; not strict raw-response compliance. audit_polish.py regenerates Figures 1–2 and fixes portable HTML/Markdown.\n\n'
text+='Figure 10: v3_run2_outputs/per_run.csv, task=modmul, instance=B_oracle (n=1), and instance=B & success=True (n=14); v3_run1_outputs/per_run.csv, task=modmul, instance=A, synth=adamw, lr=0.0003 (n=3). Means of all_acc and acc_noise0.1/0.3/0.5. The harness adds coordinate-wise Gaussian input noise using RNG seed 0 and evaluates all 12,544 pairs. Conditional successful-Claude cohort, different runs; not a controlled robustness comparison.\n\n'
for f2 in [r/'v3_run1_outputs/A_curves.csv',r/'v3_run1_outputs/per_run.csv',r/'v3_run2_outputs/per_run.csv',r/'cursor_trials/ALL30_per_trial.json',r/'final/tables/final_claude_per_run.csv']:
 text+=f'- `{f2.relative_to(r)}` SHA256 `{hashlib.sha256(f2.read_bytes()).hexdigest()}`\n'
manifest.write_text(text)
print('PASS source assertions: Cursor 27/30; API 38/45 and 92/105; local 0/12; listings exact; 10 figures; 3 tables')
