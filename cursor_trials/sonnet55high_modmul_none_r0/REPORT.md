# Cursor trial — sonnet55high_modmul_none_r0

## Provenance
- Channel: Cursor Ask (not API)
- Model (reported): Claude Sonnet 5.5 High
- Time (approx Asia/Dhaka): 2026-11-06 ~23:38–23:39
- Prompt assumed: START_HERE_modmul_none.txt (**paste not independently confirmed**)
- Prefatory: “I'll check the construction numerically before giving it to you.”
- Body: shell heredoc → `/tmp/t.py` + `python3 /tmp/t.py` (run requested; user reported no approval)

## Strict protocol
- **FAIL / not success.** Need Python-only `build_weights` block; got shell wrapper.
- process_reply(raw): ok=False success=False error_kind=name_error
- error: [name_error] generated code failed: Traceback (most recent call last):
  File "<TMPDIR>/runner.py", line 54, in <module>
  File "<generated>", line 49, in <module>
NameError: name 'EOF' is not defined


## Diagnostic (transparent extraction)
- Function body unchanged; dropped shell + self-test; restored module imports only.
- Domain: a,b∈1..112 (0 excluded), N=12544 (112²=12544)
- Shapes required: {'W1': [256, 226], 'W2': [113, 256]}
- Finite weights: True
- Shapes got: {'W1': [256, 226], 'W2': [113, 256]}
- Exact counts: {'correct': 12544, 'total': 12544, 'wrong': 0, 'accuracy': 1.0}
- Threshold 0.99 success: **True**
- Eval wall_s: 0.3939
- Margins: {'mean_margin_raw': 60.94952153916262, 'min_margin_raw': 60.949440002441406, 'mean_margin_norm': 3.8093452037856212, 'min_margin_norm': 3.8093407912752357, 'mean_prob_margin': 1.0, 'min_prob_margin': 1.0}

## Recipe (structure only; not recall vs reasoning)
Generator search + discrete log tables + Fourier features on logs (k=1..32, 4 products × 2 signs → 256 neurons).

Artifacts: `02_raw_combined_as_received.txt`, `06_build_weights_only_transparent.py`, `07_weights.npz`, `08_evaluation_report.json`
