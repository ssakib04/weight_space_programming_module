# Redaction manifest

Redactions were applied **only to the copies in this release**; the original study files are untouched. Replacement tokens are shown below. The redacted values themselves are not reproduced here.

| Token | Meaning |
|---|---|
| `<STUDY_ROOT>` | private local working-directory path of the study |
| `<PRIVATE_PATH>` | other private local path |
| `<HOME>` | home-directory path |
| `<TMPDIR>` | temporary sandbox directory in error tracebacks (not sensitive; normalized) |
| `<KAGGLE_NOTEBOOK_URL_REDACTED>` | Kaggle notebook URL containing the author's account handle (author may restore it to share links) |
| `<KAGGLE_USER_REDACTED>` | Kaggle account handle |

Notebooks also had a markdown note and a run-guard cell **prepended**. Original cells and executed outputs are otherwise unchanged.

Secret scan (`scripts/scan_secrets.py`) of the original source artifacts before copying and of this release after copying found **no** API keys, tokens, private keys, or credential values. Files only *reference* environment-variable names (e.g. `ANTHROPIC_API_KEY`, `HF_TOKEN`) read from Kaggle secrets or the environment at run time.

Because of these redactions, a few copied CSV/JSON files differ byte-for-byte from the originals hashed in `paper/SOURCE_TO_FIGURE_MANIFEST.md` (only text fields such as error tracebacks or paths changed; no numbers). `MANIFEST.sha256` hashes the release copies.

| File (release path) | Change | Count |
|---|---|---|
| `cursor_trials/ALL30_cells_summary.json` | <STUDY_ROOT> — private working-directory path | 6 |
| `cursor_trials/ALL30_per_trial.json` | <STUDY_ROOT> — private working-directory path | 36 |
| `cursor_trials/sonnet55high_modmul_none_r0/08_evaluation_report.json` | <TMPDIR> — temporary sandbox path | 1 |
| `cursor_trials/sonnet55high_modmul_none_r0/REPORT.md` | <TMPDIR> — temporary sandbox path | 1 |
| `harness/local_pilot_source/config.py` | <STUDY_ROOT> — private working-directory path | 1 |
| `notebooks/executed_records/local_pilot_phi4_mistral_v3_gated_executed.ipynb` | <KAGGLE_NOTEBOOK_URL_REDACTED> — Kaggle notebook URL containing the author's Kaggle account handle | 2 |
| `notebooks/executed_records/local_pilot_phi4_mistral_v3_gated_executed.ipynb` | <STUDY_ROOT> — private working-directory path | 1 |
| `notebooks/executed_records/local_pilot_phi4_mistral_v3_gated_executed.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/executed_records/v3_run1_executed.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/executed_records/v3_run2_executed.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/source/wsp_local_strong_pilot.ipynb` | <STUDY_ROOT> — private working-directory path | 1 |
| `notebooks/source/wsp_local_strong_pilot.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/source/wsp_modular_addition_v2.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/source/wsp_modular_addition_v3.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `notebooks/source/wsp_modular_addition_v3_run2.ipynb` | prepended markdown + guard cell — prevent accidental training, downloads or paid API calls | 1 |
| `paper/data/ALL30_cells_summary.json` | <STUDY_ROOT> — private working-directory path | 6 |
| `paper/data/ALL30_per_trial.json` | <STUDY_ROOT> — private working-directory path | 36 |
| `paper/data/local_pilot_per_run.csv` | <TMPDIR> — temporary sandbox path | 7 |
| `paper/data/v3_run1_outputs_per_run.csv` | <TMPDIR> — temporary sandbox path | 37 |
| `paper/scripts/finish_package.py` | <STUDY_ROOT> — private working-directory path | 1 |
| `paper/tables/cost_ledger.csv` | <KAGGLE_NOTEBOOK_URL_REDACTED> — Kaggle notebook URL containing the author's Kaggle account handle | 3 |
| `results/tables/cost_ledger.csv` | <KAGGLE_NOTEBOOK_URL_REDACTED> — Kaggle notebook URL containing the author's Kaggle account handle | 3 |
| `runs/local_pilot_phi4_mistral/RESULTS_REPORT.md` | <KAGGLE_NOTEBOOK_URL_REDACTED> — Kaggle notebook URL containing the author's Kaggle account handle | 1 |
| `runs/local_pilot_phi4_mistral/RESULTS_REPORT.md` | <STUDY_ROOT> — private working-directory path | 2 |
| `runs/local_pilot_phi4_mistral/manifest.json` | <KAGGLE_NOTEBOOK_URL_REDACTED> — Kaggle notebook URL containing the author's Kaggle account handle | 2 |
| `runs/local_pilot_phi4_mistral/per_run.csv` | <TMPDIR> — temporary sandbox path | 7 |
| `runs/v2_outputs/per_run.csv` | <TMPDIR> — temporary sandbox path | 45 |
| `runs/v2_outputs/results.json` | <TMPDIR> — temporary sandbox path | 45 |
| `runs/v3_replay_and_tests/replay_fakecifar.csv` | <TMPDIR> — temporary sandbox path | 19 |
| `runs/v3_replay_and_tests/replay_realcifar.csv` | <TMPDIR> — temporary sandbox path | 19 |
| `runs/v3_replay_and_tests/replay_realcifar.log` | <TMPDIR> — temporary sandbox path | 1 |
| `runs/v3_replay_and_tests/unit_results.json` | <TMPDIR> — temporary sandbox path | 10 |
| `runs/v3_replay_and_tests/unit_tests.log` | <TMPDIR> — temporary sandbox path | 10 |
| `runs/v3_run1_outputs/per_run.csv` | <TMPDIR> — temporary sandbox path | 37 |
| `runs/v3_run1_outputs/results.json` | <TMPDIR> — temporary sandbox path | 37 |

## Excluded from the release
- Credentials of any kind (none were present in the files copied).
- Full chat transcripts, HF caches, third-party model weights, LLM reply caches, duplicate zip bundles, the superseded v1 notebook, and v2 harness-extracted code (see `PROVENANCE.md`).
