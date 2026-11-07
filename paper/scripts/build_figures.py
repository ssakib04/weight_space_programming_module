#!/usr/bin/env python3
"""Build manuscript figures from audited real data only. No fabricated curves."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
MANIFEST = []

# Okabe-Ito colorblind-safe
C = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'sky': '#56B4E9', 'vermillion': '#D55E00', 'purple': '#CC79A7',
    'yellow': '#F0E442', 'black': '#000000', 'gray': '#666666',
}

def save(fig, name, source, note=''):
    png = FIG / f'{name}.png'
    pdf = FIG / f'{name}.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    MANIFEST.append({'figure': name, 'png': str(png), 'pdf': str(pdf), 'source': source, 'note': note})
    print('wrote', name)

# ---------- Fig 1: Instance A learning curves (REAL A_curves) ----------
A = pd.read_csv(OUT/'data/A_curves_run1.csv')
comp = pd.read_csv(OUT/'data/comparison_table.csv')
# Select LR with best mean val at last checkpoint for each task/opt (match paper protocol)
def best_lr_row(task, opt):
    sub = A[(A.task==task)&(A.opt==opt)]
    if sub.empty: return None
    last = sub.sort_values('step').drop_duplicates(['lr','seed'], keep='last').reset_index(drop=True)
    means = last.groupby('lr').val_acc.mean()
    lr = float(means.idxmax())
    return lr

fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.8))
axs = axs.ravel()
tasks = ['modadd','modmul','parity','permadd','cifar']
for ax, t in zip(axs, tasks):
    opts = ['adamw','sgd'] if t != 'cifar' else ['adam','sgd']
    colors = [C['blue'], C['orange']]
    for opt, col in zip(opts, colors):
        lr = best_lr_row(t, opt)
        if lr is None: continue
        g = A[(A.task==t)&(A.opt==opt)&(A.lr==lr)].groupby('step').agg(
            mean=('val_acc','mean'), std=('val_acc','std'), n=('val_acc','count')).reset_index()
        ax.plot(g.step, g['mean'], '-o', ms=3.5, color=col, label=f'{opt} lr={lr:g} (n={int(g.n.iloc[0])} seeds)')
        ax.fill_between(g.step, g['mean']-g['std'].fillna(0), g['mean']+g['std'].fillna(0), color=col, alpha=0.18)
    ax.set_title(t, fontsize=11)
    ax.set_xlabel('training step (checkpoint)')
    ax.set_ylabel('validation accuracy')
    ax.set_ylim(-0.02, 1.05)
    if t != 'cifar':
        ax.set_xscale('log')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)
axs[5].axis('off')
axs[5].text(0.0, 0.85, 'Source: v3_run1_outputs/A_curves.csv\n'
            'Protocol: SPLIT train/val/test = 0.3/0.1/0.6;\n'
            'seeds 0,1,2; checkpoints as logged.\n'
            'LR shown = argmax mean final val_acc\n'
            'among the Run-1 grid (not interpolated).\n'
            'Run 2 / local-pilot A_curves.csv empty\n'
            '(A disabled)—no fabricated curves.',
            fontsize=8.5, family='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#aaa'))
fig.suptitle('Instance A (gradient descent) validation learning curves — Run 1, real checkpoints', fontsize=12, y=1.01)
fig.tight_layout()
save(fig, 'fig01_learning_curves_A',
     'data/A_curves_run1.csv <- v3_run1_outputs/A_curves.csv',
     'Per-step val_acc at CHECKPOINTS; mean±sd over seeds at validation-selected LR')

# ---------- Fig 2: Cursor 3x2 with trial dots ----------
trials = json.loads((OUT/'data/ALL30_per_trial.json').read_text())['trials']
labels = ['Claude Sonnet 5.5 High','Gemini 4 Argon Medium','GPT-5.6 Sol Medium']
slugs = ['sonnet55high','gemini4argonmedium','gpt56solmedium']
tasks_c = ['modmul','permadd']
fig, ax = plt.subplots(figsize=(8.2, 4.2))
xpos = np.arange(len(labels))
width = 0.35
for ti, task in enumerate(tasks_c):
    rates=[]; los=[]; his=[]; dots_x=[]; dots_y=[]
    for i, (lab, slug) in enumerate(zip(labels, slugs)):
        rows=[t for t in trials if t['model_slug']==slug and t['task']==task]
        assert len(rows)==5
        k=sum(1 for t in rows if t['success'])
        n=5
        # Wilson CI
        z=1.959963984540054; ph=k/n; d=1+z*z/n; c=(ph+z*z/(2*n))/d; h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
        lo,hi=max(0,c-h),min(1,c+h)
        rates.append(ph); los.append(ph-lo); his.append(hi-ph)
        for j,t in enumerate(rows):
            dots_x.append(i + (ti-0.5)*width)
            # jitter success/fail vertically slightly for visibility of n=5
            base = 1.0 if t['success'] else 0.0
            dots_y.append(base + (j-2)*0.012)
    offs = (ti-0.5)*width
    ax.bar(xpos+offs, rates, width, yerr=[los,his], capsize=3,
           color=[C['blue'], C['green']][ti], alpha=0.85, label=f'{task}/none (Wilson 95% CI)', edgecolor='k', linewidth=0.4)
    ax.scatter([i+(ti-0.5)*width for i in xpos for _ in range(5)], 
               [dots_y[i*5:(i+1)*5] for i in range(3)] and [y for i in range(3) for y in 
                [(1.0 if trials_row['success'] else 0.0)+(j-2)*0.012 
                 for j,trials_row in enumerate([t for t in trials if t['model_slug']==slugs[i] and t['task']==task])]],
               s=18, color='k', zorder=5, alpha=0.7)
# simpler dots loop
ax.clear()
for ti, task in enumerate(tasks_c):
    rates=[]; yerr_lo=[]; yerr_hi=[]
    for i, slug in enumerate(slugs):
        rows=[t for t in trials if t['model_slug']==slug and t['task']==task]
        k=sum(1 for t in rows if t['success']); n=5
        z=1.959963984540054; ph=k/n; d=1+z*z/n; c=(ph+z*z/(2*n))/d; h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
        lo,hi=max(0,c-h),min(1,c+h)
        rates.append(ph); yerr_lo.append(ph-lo); yerr_hi.append(hi-ph)
    offs=(ti-0.5)*width
    ax.bar(xpos+offs, rates, width, yerr=[yerr_lo,yerr_hi], capsize=3,
           color=[C['blue'], C['green']][ti], alpha=0.88, label=f'{task}/none', edgecolor='k', linewidth=0.4)
    for i, slug in enumerate(slugs):
        rows=[t for t in trials if t['model_slug']==slug and t['task']==task]
        for j,t in enumerate(rows):
            y = (1.0 if t['success'] else 0.0) + (j-2)*0.015
            ax.plot(i+offs, y, 'o', ms=4.5, color='white', markeredgecolor='k', markeredgewidth=0.8, zorder=6)
ax.set_xticks(xpos)
ax.set_xticklabels(['Sonnet 5.5 High\n(user-reported)','Gemini 4 Argon Med.\n(user-reported)','GPT-5.6 Sol Med.\n(user-reported)'], fontsize=8.5)
ax.set_ylim(-0.05, 1.35)
ax.set_ylabel('Construction success rate (acc ≥ 0.99)')
ax.set_title('Cursor channel: 3 models × 2 no-hint tasks × 5 trials = 27/30 successes\n(dots = individual trials; error bars = per-cell Wilson 95% CI; not a ranking)', fontsize=10)
ax.legend(loc='upper center', ncol=2, fontsize=9)
ax.axhline(1.0, color=C['gray'], ls=':', lw=0.8)
ax.text(0.02, 1.22, 'Total 27/30; failures: sonnet modmul r3; gemini modmul r2,r4', fontsize=8, transform=ax.get_yaxis_transform())
fig.tight_layout()
save(fig, 'fig02_cursor_3x2', 'data/ALL30_per_trial.json', 'n=5 per cell; user-reported labels')

# ---------- Fig 3: API modmul hint ablation ----------
mm = pd.read_csv(OUT/'tables/final_modmul_combined.csv')
mm = mm[mm.hint!='ALL'] if 'ALL' in set(mm.hint) else mm
# file may use hint column values full/partial/none/all
mm = mm[mm.hint.str.lower().isin(['full','partial','none'])]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
hints = list(mm.hint)
rates = mm.successes/mm.n
los = rates - mm.wilson_lo
his = mm.wilson_hi - rates
ax.bar(hints, rates, color=C['blue'], edgecolor='k', width=0.55, yerr=[los,his], capsize=4)
for i,r in enumerate(mm.itertuples()):
    ax.text(i, rates.iloc[i]+0.03, f'{int(r.successes)}/{int(r.n)}', ha='center', fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Success rate (acc ≥ 0.99)')
ax.set_xlabel('Prompt hint level')
ax.set_title('API Claude (claude-sonnet-5-5) modular multiplication — combined 38/45\n(replayed v2 reps 0–9 + Run 2 reps 10–14; Wilson 95% CI)', fontsize=10)
fig.tight_layout()
save(fig, 'fig03_api_modmul_hints', 'tables/final_modmul_combined.csv', 'API channel only')

# ---------- Fig 4: API failure taxonomy ----------
fails = pd.read_csv(OUT/'tables/final_claude_failures_list.csv')
# outcome column
fig, ax = plt.subplots(figsize=(6.8, 3.6))
vc = fails['outcome'].value_counts() if 'outcome' in fails.columns else fails.iloc[:,-1].value_counts()
# also by task
tasks_f = fails['task'].value_counts()
x = np.arange(len(tasks_f))
# stack truncated vs wrong if possible
sub = fails.copy()
if 'outcome' not in sub.columns:
    sub['outcome'] = sub.iloc[:,-1]
cats = sorted(sub.outcome.unique())
bottom=np.zeros(len(tasks_f.index))
colors_c = [C['vermillion'], C['orange'], C['purple'], C['gray']]
for ci, cat in enumerate(cats):
    vals=[((sub.task==t)&(sub.outcome==cat)).sum() for t in tasks_f.index]
    ax.bar(x, vals, bottom=bottom, label=cat, color=colors_c[ci%len(colors_c)], edgecolor='k', width=0.55)
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(list(tasks_f.index))
ax.set_ylabel('Count of unsuccessful replies (v3 harness)')
ax.set_title(f'API Claude algorithmic failures under v3 harness (n_fail rows={len(fails)})\nSeparate from successes; truncations ≠ wrong weights', fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, 'fig04_api_failure_taxonomy', 'tables/final_claude_failures_list.csv')

# ---------- Fig 5: Open models validity/success ----------
pr = pd.read_csv(OUT/'data/local_pilot_per_run.csv')
pr = pr[pr.instance=='B']
# map short names
def short(s):
    if 'phi' in str(s).lower(): return 'Phi-4'
    if 'mistral' in str(s).lower(): return 'Mistral-Small-24B'
    return str(s)
pr = pr.copy()
pr['model'] = pr.synth.apply(short)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
models = ['Phi-4','Mistral-Small-24B']
tasks_o = ['modmul','permadd']
x = np.arange(len(models))
w = 0.35
for ti, task in enumerate(tasks_o):
    valid=[]; succ=[]
    for m in models:
        sub=pr[(pr.model==m)&(pr.task==task)]
        valid.append(sub.ok.sum()/len(sub) if len(sub) else 0)
        succ.append(sub.success.sum()/len(sub) if len(sub) else 0)
    ax.bar(x+(ti-0.5)*w, valid, w, label=f'{task} valid code', color=[C['sky'], C['blue']][ti], edgecolor='k', alpha=0.9)
    ax.bar(x+(ti-0.5)*w, succ, w, label=f'{task} success ≥0.99', color=[C['green'], C['orange']][ti], edgecolor='k', alpha=0.95)
# Actually overlapping bars confusing - use grouped: for each model-task show valid and success as two bars
ax.clear()
labels_x=[]; vals_v=[]; vals_s=[]
for m in models:
  for task in tasks_o:
    sub=pr[(pr.model==m)&(pr.task==task)]
    labels_x.append(f'{m}\n{task}')
    vals_v.append(sub.ok.sum())
    vals_s.append(sub.success.sum())
xpos=np.arange(len(labels_x))
ax.bar(xpos-0.18, vals_v, 0.35, label='Valid runnable code (ok)', color=C['sky'], edgecolor='k')
ax.bar(xpos+0.18, vals_s, 0.35, label='Construction success (≥0.99)', color=C['green'], edgecolor='k')
ax.set_xticks(xpos); ax.set_xticklabels(labels_x, fontsize=8)
ax.set_ylabel('Count out of n=3')
ax.set_ylim(0, 3.5)
ax.set_title('Open-weight NF4 pilot (Phi-4 & Mistral-Small-24B): 0/12 successes, 5/12 valid code\nInfra completed all 12 generations; no truncation observed under max_new_tokens=8192', fontsize=9.5)
ax.legend(fontsize=8)
for i,(v,s) in enumerate(zip(vals_v, vals_s)):
    ax.text(i-0.18, v+0.05, str(int(v)), ha='center', fontsize=8)
    ax.text(i+0.18, s+0.05, str(int(s)), ha='center', fontsize=8)
fig.tight_layout()
save(fig, 'fig05_open_models_pilot', 'data/local_pilot_per_run.csv', 'Phi/Mistral NF4 pilot only; Qwen separate caveat')

# ---------- Fig 6: CIFAR endpoints ----------
cif = pd.read_csv(OUT/'tables/final_cifar.csv')
fig, ax = plt.subplots(figsize=(8.0, 3.8))
# select key rows
rows=[]
for _,r in cif.iterrows():
    lab = f"{r.instance}: {str(r.synthesizer)[:42]}"
    rows.append((lab, r.mean_test_acc, r.uses_training_data, r.uses_gradients))
# keep a readable subset
keep_idx=[]
for i,(lab,acc,ud,ug) in enumerate(rows):
    if any(k in lab.lower() for k in ['adam','sgd','claude — run 2','claude filters','chance','qwen2.5-7b-instruct,1','qwen2.5-coder']):
        keep_idx.append(i)
# simpler manual order
order_keys = [
    ('chance', 'chance'),
    ('B,Claude — Run 2', 'B Claude pure (Run 2)'),
    ('B+ridge,Claude filters', 'B+ridge Claude filters'),
    ('B+ridge,Qwen2.5-7B', 'B+ridge Qwen-7B'),
    ('B+ridge,Qwen2.5-Coder', 'B+ridge Qwen-14B'),
    ('A,adam', 'A Adam'),
    ('A,sgd', 'A SGD'),
]
ys=[]; labs=[]; cols=[]
for key, lab in order_keys:
    hit = cif[cif.apply(lambda r: key.split(',')[0] in str(r.instance) and key.split(',')[1].lower() in str(r.synthesizer).lower() if ',' in key else key in str(r.synthesizer).lower() or key in str(r.instance).lower(), axis=1)]
    if hit.empty and key=='chance':
        ys.append(0.1); labs.append(lab); cols.append(C['gray']); continue
    if hit.empty:
        continue
    r=hit.iloc[0]
    ys.append(r.mean_test_acc); labs.append(lab)
    if r.uses_gradients: cols.append(C['blue'])
    elif r.uses_training_data: cols.append(C['orange'])
    else: cols.append(C['green'])
ax.barh(range(len(ys)), ys, color=cols, edgecolor='k')
ax.set_yticks(range(len(ys))); ax.set_yticklabels(labs, fontsize=8.5)
ax.set_xlabel('CIFAR-10 test accuracy')
ax.set_xlim(0, 0.85)
ax.axvline(0.1, color=C['gray'], ls='--', lw=0.9, label='chance 10%')
ax.set_title('CIFAR-10 exploratory endpoints (not a matched feature-control study)\nGreen=pure WSP (no data); orange=data-assisted ridge; blue=gradient-trained', fontsize=9.5)
fig.tight_layout()
save(fig, 'fig06_cifar_endpoints', 'tables/final_cifar.csv', 'No random-filter control; B+ridge uses data')

# ---------- Fig 7: conceptual synthesis vs gradient ----------
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
def box(x,y,w,h,text,color):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=color, edgecolor='k', lw=1)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=8, wrap=True)
box(0.2,1.5,2.2,1.1,'Task spec +\nfixed architecture\n(shapes, activation)', '#EEEEEE')
box(3.0,2.0,2.4,0.9,'Instance B\nLLM → build_weights()\nNumPy arrays', '#D5E8D4')
box(3.0,0.4,2.4,0.9,'Instance A\nGradients on data\nAdamW / SGD', '#DAE8FC')
box(6.2,1.5,2.2,1.1,'Same network\nevaluated on\nall inputs', '#FFF2CC')
box(8.6,1.5,1.2,1.1,'Accuracy\n≥ 0.99?', '#F8CECC')
for (x1,y1,x2,y2) in [(2.4,2.0,3.0,2.4),(2.4,1.7,3.0,0.85),(5.4,2.4,6.2,2.0),(5.4,0.85,6.2,1.7),(8.4,2.0,8.6,2.0)]:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', lw=1.2, color='k'))
ax.set_title('Conceptual comparison (not a measurement): same architecture, different weight-writing paths', fontsize=10, pad=8)
fig.tight_layout()
save(fig, 'fig07_conceptual_paths', 'conceptual diagram — not empirical data', 'Illustration only')

# ---------- Fig 8: Fourier 8-unit identity ----------
fig, ax = plt.subplots(figsize=(9.0, 3.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')
ax.set_title('Single-frequency product identity used by compact successful constructions', fontsize=11, pad=6)
ax.text(0.3, 3.3, 'one-hot(a) ‖ one-hot(b)', fontsize=9, bbox=dict(boxstyle='round', fc='#EEE'))
ax.text(3.5, 3.5, 'plus unit: u+v', fontsize=8, color=C['blue'])
ax.text(3.5, 2.9, 'minus unit: u−v', fontsize=8, color=C['vermillion'])
ax.text(6.2, 3.2, 'square\nactivation', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', fc='#FFF2CC'))
ax.text(8.3, 3.2, '((u+v)² − (u−v)²)/4 = uv', fontsize=10,
        bbox=dict(boxstyle='round', fc='#D5E8D4', ec='k'))
ax.text(0.3, 1.6, 'Choose u,v ∈ {cos θ, sin θ}\non VAL[a], VAL[b]', fontsize=8.5)
ax.text(4.0, 1.6, 'Readout vs cos/sin(θ_c)\n→ score ∝ cos(θ_a+θ_b−θ_c)', fontsize=8.5)
ax.text(8.0, 1.6, 'argmax_c selects modular\nsum (or sum of discrete logs)', fontsize=8.5,
        bbox=dict(boxstyle='round', fc='#DAE8FC', ec='k'))
ax.annotate('', xy=(3.3,3.3), xytext=(2.7,3.3), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.annotate('', xy=(6.0,3.3), xytext=(5.3,3.3), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.annotate('', xy=(8.2,3.3), xytext=(7.3,3.3), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(0.3, 0.35, 'Empirically realized as 8 connected hidden units in gpt56solmedium_permadd_none_r0 (acc=1.0 on 12,769 pairs).', fontsize=8, style='italic')
fig.tight_layout()
save(fig, 'fig08_fourier_identity', 'algebraic identity + Cursor success artifact', 'Diagram explaining math; scores from evaluation reports')

# ---------- Fig 9: replay v2 vs v3 ----------
ba = pd.read_csv(OUT/'tables/final_replay_v2_vs_v3.csv')
fig, ax = plt.subplots(figsize=(7.5, 3.6))
# expect columns task, prompt, n, v2_success, v3_success
lab = ba.apply(lambda r: f"{r.task}/{r.prompt}", axis=1)
x=np.arange(len(ba))
ax.bar(x-0.18, ba.v2_success, 0.35, label='v2 harness', color=C['vermillion'], edgecolor='k')
ax.bar(x+0.18, ba.v3_success, 0.35, label='v3 harness (same replies)', color=C['green'], edgecolor='k')
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=35, ha='right', fontsize=7.5)
ax.set_ylabel('Successes')
ax.set_title(f'Same Claude replies, different harness: {int(ba.v2_success.sum())}/90 → {int(ba.v3_success.sum())}/90\n(Harness bugs undercounted correct constructions)', fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, 'fig09_replay_v2_vs_v3', 'tables/final_replay_v2_vs_v3.csv')

# ---------- Fig 10: robustness (reuse real Run2 metrics) ----------
# from per_run in run2 if available
r2 = pd.read_csv(OUT/'data/v3_run2_outputs_per_run.csv')
fig, ax = plt.subplots(figsize=(6.8, 3.6))
sig = [0.0, 0.1, 0.3, 0.5]
o = r2[(r2.task=='modmul')&(r2.instance=='B_oracle')]
cs = r2[(r2.task=='modmul')&(r2.instance=='B')&(r2.success==True)]
# AdamW from run1 comparison at end — noise cols on per_run run1
r1 = pd.read_csv(OUT/'data/v3_run1_outputs_per_run.csv')
aa = r1[(r1.task=='modmul')&(r1.instance=='A')&(r1.synth=='adamw')&(r1.lr==0.0003)]
def mean_curve(df):
    ys=[df['all_acc'].mean()]
    for s in [0.1,0.3,0.5]:
        col=f'acc_noise{s}'
        ys.append(df[col].mean() if col in df.columns else np.nan)
    return ys
if len(o):
    ax.plot(sig, mean_curve(o), 'k-o', label='oracle')
if len(cs):
    ax.plot(sig, mean_curve(cs), '-o', color=C['blue'], label=f'Claude successes Run2 (n={len(cs)})')
if len(aa):
    ax.plot(sig, mean_curve(aa), '-o', color=C['green'], label=f'AdamW Run1 (n={len(aa)})')
ax.set_xlabel('Gaussian noise σ on one-hot input')
ax.set_ylabel('Accuracy')
ax.set_title('Robustness on modmul (real logged noise evaluations)', fontsize=10)
ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.25)
fig.tight_layout()
save(fig, 'fig10_robustness_modmul', 'v3_run2_outputs/per_run.csv + v3_run1_outputs/per_run.csv')

# Write manifest
import json
(OUT/'SOURCE_TO_FIGURE_MANIFEST.json').write_text(json.dumps(MANIFEST, indent=2))
md_lines=['# Source → figure manifest\n','All figures built by `scripts/build_figures.py` from audited files. No interpolated training curves.\n']
for m in MANIFEST:
    md_lines.append(f"- **{m['figure']}**: source `{m['source']}` — {m['note']}\n")
md_lines.append('\n## Training time-series availability\n')
md_lines.append('- **Present:** `v3_run1_outputs/A_curves.csv` (468 rows; tasks modadd/modmul/parity/permadd/cifar; opts adamw/sgd/adam; seeds 0–2; checkpoint steps).\n')
md_lines.append('- **Empty / unused:** `v3_run2_outputs/A_curves.csv` (A disabled in Run 2); local-pilot `A_curves.csv` (A not run in that pilot). **No curves fabricated for those runs.**\n')
(OUT/'SOURCE_TO_FIGURE_MANIFEST.md').write_text(''.join(md_lines))
print('MANIFEST', len(MANIFEST), 'figures')
