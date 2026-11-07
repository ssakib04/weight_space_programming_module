# How Reliably Can Language Models Construct Neural Network Weights? Evidence from Modular Arithmetic Tasks

**Author:** Kazi Sadman Sakib

> **PRE-PUBLICATION RELEASE CANDIDATE.** Prepared locally on 7 Nov 2026 (Asia/Dhaka) for author review. It has **not** been posted to GitHub or anywhere else, and the paper has **not** been submitted or peer reviewed. The license is **not decided** (see `LICENSE_PENDING.md`). Please don't redistribute until the author approves a public release.

This repository contains the paper and the artifacts behind it: frozen prompts, the original model responses, the weights each response produced, per-trial scores, sanitized notebooks, and an offline evaluator that re-scores every saved construction **without API calls, network access, or training**.

## Main results (each channel reported separately and never pooled)

| Channel | Task(s) | Construction success (accuracy ≥ 0.99 on the full domain) |
|---|---|---|
| Cursor Ask (no-hint, n=5 per cell) | modmul / permadd | Claude Sonnet 5.5 High 4/5, 5/5 · Gemini 4 Argon Medium 3/5, 5/5 · GPT-5.6 Sol Medium 5/5, 5/5 → **27/30** |
| Anthropic API, `claude-sonnet-5-5` | modmul (3 hint levels) | **38/45** (full 12/15, partial 13/15, none 13/15) |
| Anthropic API, `claude-sonnet-5-5` | modadd + modmul + parity | **92/105** |
| Local open weights, NF4 (Phi-4, Mistral-Small-24B) | modmul / permadd | **0/12** (5 runnable, 7 runtime/shape failures) |
| Local open weights, 4-bit (Qwen2.5-7B / Coder-14B), earlier run | 4 algorithmic tasks | **0/72**, with a 2,048-token cap, so not budget-matched to the 8,192-token pilot |

These are exploratory reliability estimates from small samples. They are not a head-to-head ranking. Cursor model names are the labels shown in Cursor's model picker. Cursor session conditions (Ask mode, unchanged prompts, empty workspace, a separate chat per query) are an **author attestation recorded on 7 Nov 2026** (`cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json`), not independently verified. In one Cursor trial (`sonnet55high_modmul_none_r0`) the response arrived inside a shell-wrapper. It counts as a construction success only after a documented, non-numerical extraction step.

## Quickstart (offline, about 1 minute on CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy + pandas (+ matplotlib for the optional figure rebuild)
python scripts/verify_release.py         # all checks; writes verification_output/
python scripts/verify_release.py --skip-figures
```

`verify_release.py` checks the SHA-256 manifest and the frozen-prompt hashes. It re-scores all 30 Cursor weight files in float32 and float64 and checks them against the saved reports, asserting 27/30 and the per-cell counts. It recounts the API results (92/105, 38/45) and re-scores the saved Run-2 API weights. It recounts the open-model results (0/12 with 5 runnable; Qwen 0/72) and re-scores their saved weights. It also runs the secret/private-detail scan and rebuilds the 10 figures from the packaged data. **It never executes model-generated code.** `harness/wsp_eval.py` loads only the saved `.npz` arrays.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript v3 (PDF, HTML, Markdown), figures (PNG/PDF), figure data, `SOURCE_TO_FIGURE_MANIFEST.md`, review notes |
| `prompts/cursor_frozen/` | Byte-frozen Cursor prompts (`START_HERE_*.txt`), their SHA-256 manifest, protocol freeze, and run instructions |
| `prompts/open_models_phi4_mistral/` | Exact chat messages used for the Phi-4 / Mistral pilot |
| `prompts/api_and_run_logs/` | Every unique system/user prompt extracted verbatim from the run logs (API and Qwen runs) |
| `cursor_trials/<trial>/` | Each of the 30 trials: raw response as received, exact extracted function, metadata, saved weights (`05_weights.npz`; `07_weights.npz` for the wrapper case), evaluation report, generated code |
| `runs/v2_outputs/` | Original v2 API and local logs (`llm_calls.jsonl` holds the full responses) and scores. These replies were later re-scored with the v3 harness |
| `runs/v3_run1_outputs/` | Run 1: gradient baselines (`A_curves.csv`), Qwen responses, scores, generated code and weights |
| `runs/v3_run2_outputs/` | Run 2: fresh API responses (`llm_calls.jsonl`), scores, generated code and weights |
| `runs/v3_replay_and_tests/` | Offline v3-harness replay of the v2 replies, plus unit-test logs |
| `runs/local_pilot_phi4_mistral/` | Phi-4 / Mistral NF4 pilot: per-run JSON, scores, generated code, environment manifest |
| `results/tables/` | Consolidated per-channel tables used in the paper |
| `harness/` | `wsp_eval.py` (offline NumPy evaluator), the original pipeline source `v3_src/` with its `tests/`, and the local-pilot source |
| `notebooks/executed_records/` | Sanitized executed notebooks whose outputs produced the reported runs |
| `notebooks/source/` | Sanitized source notebooks |
| `scripts/` | `verify_release.py`, `scan_secrets.py` |
| `PROVENANCE.md` | Models, revisions, settings, dates, and what is or isn't verified |
| `ENVIRONMENT.md` | Software versions for the original runs and for offline verification |
| `REDACTION_MANIFEST.md` | Every change made to the copies in this release |
| `MANIFEST.sha256` | Hash of every file |

## Re-running the original pipelines (optional; may cost money)

The notebooks are records, and **nothing re-runs by default**. A guard cell at the top of each one stops it unless you set `WSP_ALLOW_NOTEBOOK_RUN=1`. Without `WSP_PAID_API_OPT_IN=1`, the v3 notebooks force the Anthropic API off (`WSP_RUN_CLAUDE=0`, dry-run/cache only). The historical v2 notebook refuses to run at all unless both variables are set, because it can call the paid API whenever a key is available. Re-running can take GPU hours, download about 90 GiB of model weights (local pilot), or incur API charges. You'd need your own credentials, and none are included.

## Citation

See `CITATION.cff`. It contains the title and author only. No DOI or repository URL exists yet.
