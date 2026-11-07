# Environment

## Offline verification (this release)
Python 3.13.5; numpy 2.5.4; pandas 3.0.6 / 3.1.1; matplotlib 3.11.2 (figures only). CPU only. No network.

## Original runs (as logged)
| Run | Platform | Key versions | Source of the record |
|---|---|---|---|
| v3 Run 1 and Run 2 (Kaggle notebooks) | Kaggle GPU | torch 2.11.0+cu130 | `notebooks/executed_records/v3_run*_executed.ipynb` (startup line) |
| Phi-4 / Mistral NF4 pilot | Kaggle, 2×T4 | torch 2.11.0+cu130 (CUDA 13.0), transformers 4.51.3, accelerate 1.6.0, bitsandbytes 0.50.2, mistral-common 1.5.4 | `runs/local_pilot_phi4_mistral/manifest.json`, `GPU_PREFLIGHT_READY.json` |
| Cursor trial scoring | local box, CPU | torch float32 harness, plus NumPy float32 and float64 cross-checks | `cursor_trials/*/0[68]_evaluation_report.json` |

The reported Cursor scores come from the float32 harness. `harness/wsp_eval.py` reproduces every float32 count exactly and every float64 count that was recorded. For the failed trial `gemini4argonmedium_modmul_none_r4`, float32 gives 3,126 and float64 gives 4,904 of 12,544; it fails in both.
