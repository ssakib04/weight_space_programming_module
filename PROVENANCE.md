# Provenance

All times are Asia/Dhaka (UTC+6) unless a record carries its own zone.

## Cursor channel (30 trials)
- **Models:** labels as shown in Cursor's model picker and reported by the author: "Claude Sonnet 5.5 High", "Gemini 4 Argon Medium", "GPT-5.6 Sol Medium". **Not provider-verified.** Cursor does not expose provider model versions or sampling settings in this channel.
- **Prompts:** `prompts/cursor_frozen/START_HERE_modmul_none.txt` and `START_HERE_permadd_none.txt`, byte-frozen with SHA-256 values in `prompts_start_here_manifest.json` (dated 2026-11-07 00:04). The freeze was declared **after** the first three modmul r0 trials (`PROTOCOL_FREEZE.md`). It is a post-pilot freeze, not a preregistration.
- **Session conditions:** the author attests that every query used Ask mode, the unchanged prompt, an empty workspace, and a separate chat (recorded 2026-11-07 13:20; `cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json`). This was **not independently verified**. Per-trial metadata written at collection time (e.g. `mode_fresh_chats_newly_confirmed: false`, `exact_paste_confirmed: false`) is left unchanged on purpose.
- **Endpoint:** construction success = accuracy ≥ 0.99 on the full domain (12,544 modmul pairs; 12,769 permadd pairs). 27/30. `sonnet55high_modmul_none_r0` fails the strict Python-only path because of a shell heredoc wrapper. Its unchanged `build_weights` function, extracted transparently, scores 12,544/12,544 (`04_extraction_strict.json`, `06_extraction_notes.json`). No numerical repair was applied to any trial.

## Anthropic API channel
- Model requested and returned: `claude-sonnet-5-5`. The API rejected temperature 0.0, so default sampling was used. `max_tokens` = 6000 for algorithmic tasks (v2 and Run 2) and 12000 for CIFAR (Run 2).
- Reps 0–9 are v2 replies (`runs/v2_outputs/llm_calls.jsonl`), re-scored offline with the v3 harness (`runs/v3_replay_and_tests/replay_realcifar.csv`); this cost nothing and made no new calls. Reps 10–14 for modmul are fresh Run-2 calls (`runs/v3_run2_outputs/llm_calls.jsonl`). Four modmul replies were truncated at the token cap, and these are counted as failures.
- Per-call cost estimates in the logs come from token counts. No account or billing identifiers are included.

## Local open-weight channel
- **Phi-4 / Mistral pilot:** `microsoft/phi-4` @ `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`; `mistralai/Mistral-Small-24B-Instruct-2501` @ `9527884be6e5616bdd54de542f9ae13384489724`. NF4 4-bit, `max_new_tokens`=8192, `do_sample=True`, temperature 0.6, top_p 0.95 (identical across models, deliberately), 3 reps × 2 tasks × 2 models.
- **Qwen (v3 Run 1):** `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-Coder-14B-Instruct`, 4-bit, `LOCAL_MAX_NEW_TOKENS`=2048, temperature 0.7, seeded sampling (seed = 1000 + rep). Model revisions were not pinned in Run 1.
- **v2 note:** in `runs/v2_outputs/llm_calls.jsonl`, the slot configured as `llama-3.1-8b` actually loaded `Qwen/Qwen2.5-7B-Instruct`. The records label rows by the model that actually loaded, and this silent fallback motivated the v3 "no silent fallback" rule.

## Gradient baselines (Instance A)
`runs/v3_run1_outputs/A_curves.csv` (468 checkpoint rows) and `per_run.csv`. The source config is in `harness/v3_src/01_config.py`.

## What was not included
- Full Cursor chat transcripts (only the responses and extracted code per trial), HF model caches and third-party model weights, LLM reply caches (duplicates of `llm_calls.jsonl`), duplicate zip bundles, the superseded v1 notebook, and v2 harness-extracted code (superseded; the original v2 responses are in `llm_calls.jsonl`).
- CIFAR-10 data, which is downloaded by the pipeline only if a notebook is explicitly re-run.
