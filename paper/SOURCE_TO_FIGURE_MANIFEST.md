# Source → figure manifest

Built by `scripts/build_figures.py` from audited files. **No fabricated or interpolated training curves.**

- **fig01_learning_curves_A**: `data/A_curves_run1.csv` — mean±sd over 3 seeds at val-selected LR
- **fig02_cursor_3x2**: `data/ALL30_per_trial.json` — 27/30 audited
- **fig03_api_modmul_hints**: `tables/final_modmul_combined.csv` — 
- **fig04_api_failure_taxonomy**: `tables/final_claude_failures_list.csv` — 
- **fig05_open_models_pilot**: `data/local_pilot_per_run.csv` — 
- **fig06_cifar_endpoints**: `tables/final_cifar.csv` — 
- **fig07_conceptual_paths**: `conceptual` — not empirical
- **fig08_fourier_identity**: `algebra + Cursor artifact` — 
- **fig09_replay_v2_vs_v3**: `tables/final_replay_v2_vs_v3.csv` — 
- **fig10_robustness_modmul**: `v3_run1/2 per_run.csv` — 

## Training time-series inventory

| Path | Status |
|------|--------|
| `v3_run1_outputs/A_curves.csv` | **Used** — 468 rows, real checkpoints |
| `v3_run2_outputs/A_curves.csv` | Empty (Instance A off) — not plotted |
| local-pilot `A_curves.csv` | Empty / unused — not plotted |
| `v2_outputs/A_curves.csv` | Available but superseded by Run 1 for main curves |

## Independent audit details

Figure 1: exact byte-identical copy of v3_run1_outputs/A_curves.csv, 468 rows. LR chosen independently per task/optimizer by maximum mean final validation accuracy across seeds 0–2; curves are checkpoint validation means ± sample SD (not CI). No test-based selection. Algorithmic split is 30/10/60; CIFAR is 45k/5k plus official 10k test. Lines join checkpoints only.

Figure 2: construction endpoint includes the documented Sonnet modmul r0 formatting-only extraction; not strict raw-response compliance. audit_polish.py regenerates Figures 1–2 and fixes portable HTML/Markdown.

Figure 10: v3_run2_outputs/per_run.csv, task=modmul, instance=B_oracle (n=1), and instance=B & success=True (n=14); v3_run1_outputs/per_run.csv, task=modmul, instance=A, synth=adamw, lr=0.0003 (n=3). Means of all_acc and acc_noise0.1/0.3/0.5. The harness adds coordinate-wise Gaussian input noise using RNG seed 0 and evaluates all 12,544 pairs. Conditional successful-Claude cohort, different runs; not a controlled robustness comparison.

- `v3_run1_outputs/A_curves.csv` SHA256 `7899528e929f3f8cc0ff7f6bd1fd5f73021d907edfdb53ea0592b71ad599098b`
- `v3_run1_outputs/per_run.csv` SHA256 `3eb3c61ac3472647b4b2bcd2d0508b9ec7597a3262592f6c5bc587992d5df657`
- `v3_run2_outputs/per_run.csv` SHA256 `0a8c2f15d29b98bc3b95fa567a08ca59bce36d89ff39ff273df8d233c04e6667`
- `cursor_trials/ALL30_per_trial.json` SHA256 `73ea1b4dba1bfb03b84155e0f13a5b71d1583323fe894068fa3a169f8ecdf21c`
- `final/tables/final_claude_per_run.csv` SHA256 `d5e4494c9ffec4b62e0cbfae836a57488063032748cb03d8356e824adb3eeb3b`

## Reader-facing figure order (v3)

- Figure 1: `figures/fig01_learning_curves_A.png`
- Figure 2: `figures/fig02_cursor_3x2.png`
- Figure 3: `figures/fig03_api_modmul_hints.png`
- Figure 4: `figures/fig04_api_failure_taxonomy.png`
- Figure 5: `figures/fig09_replay_v2_vs_v3.png`
- Figure 6: `figures/fig10_robustness_modmul.png`
- Figure 7: `figures/fig05_open_models_pilot.png`
- Figure 8: `figures/fig07_conceptual_paths.png`
- Figure 9: `figures/fig08_fourier_identity.png`
- Figure 10: `figures/fig06_cifar_endpoints.png`

v3 moved CIFAR-10 (fig06_cifar_endpoints) to Appendix B, so it is now Figure 10. Figure files and sources are unchanged from the audited v2 build.
