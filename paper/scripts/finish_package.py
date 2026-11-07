from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify
import shutil,json
p=Path(__file__).resolve().parents[1];r=p.parent
s=BeautifulSoup((p/'WSP_manuscript_v2.html').read_text(),'html.parser')
# Sequential reader-facing numbering; stable filenames continue to identify the source figures.
if not s.body.get('data-sequential-figures'):
 mapping={}
 for n,fig in enumerate(s.find_all('figure'),1):
  cap=fig.find('figcaption');old=int(cap.get_text().split('.')[0].split()[-1]);mapping[old]=n
  cap.string=cap.get_text().replace(f'Figure {old}.',f'Figure {n}.',1)
 for node in list(s.find_all(string=True)):
  text=str(node)
  for old,new in mapping.items():text=text.replace(f'Fig.\xa0{old})',f'Fig.\xa0X{new})').replace(f'Fig. {old})',f'Fig. X{new})')
  if text!=str(node):node.replace_with(text.replace('Fig.\xa0X','Fig.\xa0').replace('Fig. X','Fig. '))
 s.body['data-sequential-figures']='true'
(p/'WSP_manuscript_v2.html').write_text(str(s))
(p/'WSP_manuscript_v2.md').write_text('---\nstatus: REVIEW-READY DRAFT\nauthor: Kazi Sadman Sakib\n---\n\n'+markdownify(str(s.body),heading_style='ATX',code_language='python'))
for run in ['v3_run1_outputs','v3_run2_outputs']:shutil.copyfile(r/run/'per_run.csv',p/'data'/f'{run}_per_run.csv')
shutil.copytree(r/'v3_src',p/'source_protocol',dirs_exist_ok=True)
# Portable figure build paths. Original raw data are unchanged.
f=p/'scripts/build_figures.py';text=f.read_text().replace("ROOT = Path('<STUDY_ROOT>')\nOUT = ROOT / 'final_manuscript'","OUT = Path(__file__).resolve().parents[1]\nROOT = OUT.parent")
text=text.replace("ROOT/'v3_run2_outputs/per_run.csv'","OUT/'data/v3_run2_outputs_per_run.csv'").replace("ROOT/'v3_run1_outputs/per_run.csv'","OUT/'data/v3_run1_outputs_per_run.csv'")
f.write_text(text)
with (p/'SOURCE_TO_FIGURE_MANIFEST.md').open('a') as f:
 f.write('\n## Reader-facing figure order\n\n')
 for n,fig in enumerate(s.find_all('figure'),1):f.write(f'- Figure {n}: `{fig.img["src"]}`\n')
 f.write('\nNoise source CSVs are also included under data/v3_run1_outputs_per_run.csv and data/v3_run2_outputs_per_run.csv; evaluation and training source is included under source_protocol/.\n')
