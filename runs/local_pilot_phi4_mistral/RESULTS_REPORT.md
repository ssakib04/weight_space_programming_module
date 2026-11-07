# WSP local strong pilot v3-gated — results report

- Written: 2026-11-06T13:10:56.013414+06:00 (Asia/Dhaka)
- Notebook: <KAGGLE_NOTEBOOK_URL_REDACTED>
- Version: Version 1 of 1, scriptVersionId=`366184538`
- Kaggle status: **Successfully ran** (~2026.9s / notebook runtime ~33m 47s · GPU T4 x2)
- Launch (Asia/Dhaka): ~12:24:31 Nov 6 2026; NF4 CUDA OK 12:26:23; both preflight OK by 12:38:04; DONE ~12:58
- Artifacts: `<STUDY_ROOT>/local_pilot/outputs/v3_gated/`
- Repair source: `repair_v2/wsp_local_strong_pilot_v2_gated.ipynb`
- Prior overnight failure (bitsandbytes CPU): scriptVersionId=366072290 — repaired path succeeded here

## Verdict

**Infrastructure succeeded. Scientific pilot completed all 12 intended responses. Model solutions failed the accuracy bar.**

Do not read “Successfully ran” as scientific success. Oracles scored 1.0; LLM generations scored near chance / failed code execution. n=3 per cell is pilot evidence only, not publication-strength.

## Stack and models

- torch `2.11.0+cu130`, transformers `4.51.3`, bitsandbytes `0.50.2`, accelerate `1.6.0`
- CUDA torch `13.0`; NF4 quantize/dequantize OK at 2026-11-06T12:26:23.530094+06:00
- Sampling (manifest): do_sample=True, T=0.6, top_p=0.95, max_new_tokens=8192
- Note: logs also showed transformers warnings that `do_sample=False` with temperature set during preflight short generate; pilot rows all stopped `end_turn`.

- **microsoft/phi-4** revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`: load_generate_ok=True, footprint_bytes=8871782656, device_map_n=44, preflight reply='PREFLIGHT_OK'
- **mistralai/Mistral-Small-24B-Instruct-2501** revision `9527884be6e5616bdd54de542f9ae13384489724`: load_generate_ok=True, footprint_bytes=13800089856, device_map_n=44, preflight reply='PREFLIGHT_OK'
- both_ok gate: True

## Raw counts (model rows only; oracles excluded)

| Metric | Count |
|---|---|
| Intended / completed generations | 12 / 12 |
| Valid evaluated code (`ok=True`) | 5 / 12 |
| Successes (`success=True`) | 0 / 12 |
| Accuracy ≥ 0.99 (`all_acc`) | 0 / 12 |
| Truncations (`extract_truncated=True`) | 0 / 12 |
| Incomplete (`incomplete=True`) | 0 / 12 |
| stop_reason=end_turn | 12 / 12 |
| Generated-code failures (error_kind set) | 7 / 12 (runtime=4, shape=3) |
| Infrastructure failures (no usable model reply / non-end_turn) | 0 / 12 |

### Separate failure buckets (do not pool)

1. **Infrastructure**: 0. Preflight both models OK; MODEL_READY×2; all 12 replies local `end_turn`. (Log noise: pip dependency-conflict ERROR lines and “Could not find cuda drivers… GPU will not be used” warnings appeared, but NF4 CUDA check and GPU preflights still passed.)
2. **Truncation / incomplete**: 0. No `extract_truncated`; no `incomplete`; no length-cap stops observed on these 12.
3. **Valid-code but wrong weights**: 5 rows with `ok=True` and `success=False`; accuracies ≈ 0.000–0.0089 (far below 0.99).
4. **Generated-code runtime/shape failures**: 7 rows — code extracted and compiled (`extract_compiles=True`) but failed when building/evaluating weights.

## Per cell (from pilot_summary.csv)

| Model | Task | Prompt | n | valid | successes | mean_acc | incomplete |
|---|---|---|---:|---:|---:|---:|---:|
| microsoft/phi-4 | modmul | none | 3 | 2 | 0 | 0.005926 | 0 |
| microsoft/phi-4 | permadd | none | 3 | 1 | 0 | 0.002924 | 0 |
| mistralai/Mistral-Small-24B-Instruct-2501 | modmul | none | 3 | 1 | 0 | 0.000000 | 0 |
| mistralai/Mistral-Small-24B-Instruct-2501 | permadd | none | 3 | 1 | 0 | 0.002950 | 0 |

## Per-run detail (model rows)

| Model | Task | Rep | ok | success | all_acc | out_tok | latency_s | error_kind | error (short) |
|---|---|---:|---|---|---:|---:|---:|---|---|
| microsoft/phi-4 | modmul | 0 | True | False | 0.008849 | 716.0 | 61.12521 |  |  |
| microsoft/phi-4 | permadd | 0 | True | False | 0.008771 | 1062.0 | 98.80917 |  |  |
| microsoft/phi-4 | modmul | 1 | False | False | 0.000000 | 729.0 | 62.62988 | runtime | [runtime] generated code failed: Traceback (most recent call last): |
| microsoft/phi-4 | permadd | 1 | False | False | 0.000000 | 1039.0 | 96.67262 | runtime | [runtime] generated code failed: Traceback (most recent call last): |
| microsoft/phi-4 | modmul | 2 | True | False | 0.008929 | 699.0 | 59.43037 |  |  |
| microsoft/phi-4 | permadd | 2 | False | False | 0.000000 | 994.0 | 91.78114 | shape | [shape] generated code failed: Traceback (most recent call last): |
| mistralai/Mistral-Small-24B-Instruct-2501 | modmul | 0 | False | False | 0.000000 | 267.0 | 34.04584 | shape | [shape] generated code failed: Traceback (most recent call last): |
| mistralai/Mistral-Small-24B-Instruct-2501 | permadd | 0 | False | False | 0.000000 | 783.0 | 103.7594 | runtime | [runtime] generated code failed: Traceback (most recent call last): |
| mistralai/Mistral-Small-24B-Instruct-2501 | modmul | 1 | True | False | 0.000000 | 328.0 | 41.45283 |  |  |
| mistralai/Mistral-Small-24B-Instruct-2501 | permadd | 1 | True | False | 0.008850 | 711.0 | 93.69286 |  |  |
| mistralai/Mistral-Small-24B-Instruct-2501 | modmul | 2 | False | False | 0.000000 | 173.0 | 22.27900 | runtime | [runtime] generated code failed: Traceback (most recent call last): |
| mistralai/Mistral-Small-24B-Instruct-2501 | permadd | 2 | False | False | 0.000000 | 714.0 | 94.71526 | shape | [shape] generated code failed: Traceback (most recent call last): |

## Oracles

- modmul: ok=True success=True acc=1.0
- permadd: ok=True success=True acc=1.0

## Artifact inventory

All under `<STUDY_ROOT>/local_pilot/outputs/v3_gated/`:

| Path | SHA256 |
|---|---|
| `wsp-local-strong-pilot-v3-gated.ipynb` | `76ff01cc602f45815aace1f3bc4b5b55b8f3a8cf0e7c3edac3e3b5d3a7cbdd0e` |
| `download.txt` | `9f33d1916261d07dfdf7ddb40e045a6527aa974204f5401b91d7744937ceca7e` |
| `results.zip` | `269cb2da3dab34b681065892a60025d10c90c5d38e61ea806a97c556adb52aff` |
| `local_strong_pilot_v2_gated/per_run.csv` | `b0d1bb810e1c3d0f0fa00159856ad1578e98857d772b0ba3d04cb8946ba8b494` |
| `local_strong_pilot_v2_gated/pilot_summary.csv` | `c1664eb404e1588f39b98884da774a4271e2aaeceacea938a8900a2a48d9275c` |
| `local_strong_pilot_v2_gated/events.jsonl` | `7aaa436048b256cdc9f9f051f356a15d61197dbe4fffc3065a0701a72267d1e3` |
| `local_strong_pilot_v2_gated/GPU_PREFLIGHT_READY.json` | `f15ca9920ada127119eddb076cf26717b73de4bf8e04a69827078d0405711ff0` |
| `local_strong_pilot_v2_gated/nf4_cuda_op.json` | `afb9132be6fafb111938ceccbda1841d55eb577cd485e475f697cfb81167665e` |
| `local_strong_pilot_v2_gated/manifest.json` | `fb2b98e44c0d27f506c40b1b3cab05927207ca828dec158fb29f6f092717e1af` |

Also present: 12 response JSON files, 12+ generated_code `.py`/`.npz`, `A_curves.csv`, `C_edits.csv`, `results_partial.json`, `nvidia_smi.txt`, both `preflight_*.json`, and `results.zip` (original Kaggle Output zip).

## Constraints honored

- Free local Kaggle only; cost_usd=0 on all rows; no Claude/paid API calls.
- No automatic relaunch; no model/prompt substitution.
- Cases limited to modmul/none + permadd/none, reps 0–2 (12 intended).
- Finite monitor deadline Nov 6 19:00 Asia/Dhaka; this report lands before deadline after natural completion.

## Bottom line for Sakib

The repaired v3-gated commit fixed the overnight bitsandbytes/CUDA infra blocker. Both Phi-4 and Mistral-Small-24B loaded and generated on T4×2 under NF4. All 12 no-hint pilot responses finished. **0/12 met success or ≥0.99 accuracy**; 5/12 had valid evaluable code with near-zero accuracy; 7/12 had generated-code runtime/shape failures. Treat as a completed negative pilot on these two models under the pinned no-hint setup — not as an unfinished run.
