"""Focused independent audit corrections; no experiments or raw data edits."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from markdownify import markdownify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
R=P.parent
# Rebuild the source-traceable learning curve only, with the split label scoped correctly.
src=(P/'scripts/build_figures.py').read_text().split('# ---------- Fig 2:')[0]
src=src.replace('Protocol: SPLIT train/val/test = 0.3/0.1/0.6;', 'Algo split train/val/test = 0.3/0.1/0.6;')
exec(compile(src,'build_figures.py [fig01]','exec'))
# Rebuild Figure 2 without annotation/legend collision or vertical jitter beyond 0/1.
t=json.loads((P/'data/ALL30_per_trial.json').read_text())['trials']
fig,ax=plt.subplots(figsize=(8.2,4.2))
for ti,task in enumerate(['modmul','permadd']):
 for i,slug in enumerate(['sonnet55high','gemini4argonmedium','gpt56solmedium']):
  rows=[x for x in t if x['model_slug']==slug and x['task']==task];n=len(rows);k=sum(x['success'] for x in rows);ph=k/n;z=1.959963984540054;d=1+z*z/n;c=(ph+z*z/(2*n))/d;h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d;x=i+(ti-.5)*.35
  ax.bar(x,ph,.35,color=['#0072B2','#009E73'][ti],edgecolor='k',linewidth=.4,yerr=[[ph-(c-h)],[(c+h)-ph]],capsize=3,label=task+'/none' if i==0 else None)
  ax.scatter(x+np.linspace(-.07,.07,n),[int(q['success']) for q in rows],s=19,color='white',edgecolor='black',linewidth=.7,zorder=4)
ax.set_xticks(range(3),['Sonnet 5.5 High\n(user-reported)','Gemini 4 Argon Med.\n(user-reported)','GPT-5.6 Sol Med.\n(user-reported)'],fontsize=8.5)
ax.set_ylim(-.06,1.22);ax.set_ylabel('Construction success rate (accuracy ≥ 0.99)')
ax.set_title('Cursor: 27/30 construction successes (n=5/cell)\nDots=trials; bars=Wilson 95% CI — exploratory, not a ranking',fontsize=10)
ax.legend(loc='upper center',ncol=2,fontsize=8,frameon=False)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(P/f'figures/fig02_cursor_3x2.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
f=P/'WSP_manuscript_v2.html';s=BeautifulSoup(f.read_text(),'html.parser')
# Portable assets and compact presentation-only table rounding.
for img in s.find_all('img'):img['src']='figures/'+Path(img['src']).name
api=s.find_all('table')[2]
headers=['Hint','Replay','Fresh','Successes','n','95% CI low','95% CI high','Truncated','Wrong']
for th,name in zip(api.find_all('th'),headers):th.string=name
for row in api.find('tbody').find_all('tr'):
 cells=row.find_all('td')
 for j in [5,6]:cells[j].string=f'{float(cells[j].get_text()):.3f}'
# Clarify train/validation/test selection and no-test-tuning from real source.
for td in s.find_all('td'):
 if td.get_text()=='SPLIT (train,val,rest=test)':td.string='Algorithmic split (train,val,test)'
if not s.find(id='audit-methods'):
 p=s.new_tag('p',id='audit-methods');p.string=('Algorithmic Instance A uses full-batch cross-entropy, gradient clipping at norm 1, and seed-specific random splits. For each task/optimizer, Figure 1 selects one learning rate by highest mean validation accuracy across seeds at the final logged checkpoint, then displays all logged validation checkpoints (sample SD, not a confidence interval). Test accuracy does not select the learning rate. Lines connect measured checkpoints; no intermediate measurements are implied. CIFAR instead uses 45,000 training and 5,000 validation images from the official training set and the official 10,000-image test set; batch size 128, random horizontal flips, Adam or SGD (momentum 0.9, weight decay 5e-4), and the same final-validation LR rule. Inputs use channel means (0.4914,0.4822,0.4465) and SDs (0.2470,0.2435,0.2616). The ridge diagnostic uses 10,000 training images, standardized frozen features and ridge parameter 1.0 (see v3_src/60_cifar.py).')
 s.find_all('table')[0].insert_after(p)
for cap in s.find_all('figcaption'):
 text=cap.get_text()
 if 'fig10_robustness_modmul' in cap.parent.find('img')['src']:
  cap.string=(text.split('.')[0]+'. Exploratory logged modmul input-noise diagnostic: means over 14 successful fresh Run-2 Claude constructions (selected on clean accuracy), one Run-2 analytic oracle, and three Run-1 AdamW seeds at LR 0.0003. Independent Gaussian noise is added to all input coordinates at σ=0.1,0.3,0.5, with evaluation RNG seed 0; σ=0 is clean all-domain accuracy. All 12,544 inputs are scored at each level. Curves are conditional, cross-run diagnostics, not matched robustness or model-ranking evidence; no new evaluations were run for this figure.')
 if text.startswith('Figure 1.'):
  cap.string=text.replace('No interpolation.', 'Only logged checkpoints; lines are visual guides. The 30/10/60 split applies to algorithmic tasks, not CIFAR.')
# Avoid a dangling closing footer on an otherwise blank page.
for foot in s.select('.foot'):foot.decompose()
html=str(s).replace('accuracy on every input','accuracy across the complete input domain')
f.write_text(html)
(P/'WSP_manuscript_v2.md').write_text('---\nstatus: REVIEW-READY DRAFT\nauthor: Kazi Sadman Sakib\n---\n\n'+markdownify(str(s.body),heading_style='ATX',code_language='python'))
