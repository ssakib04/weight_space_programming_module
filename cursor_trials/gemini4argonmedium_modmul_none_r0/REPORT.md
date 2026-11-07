# Cursor trial — gemini4argonmedium_modmul_none_r0

## Provenance
- Model (reported): Gemini 4 Argon Medium
- Mode: Ask instructed; not independently reconfirmed
- Time approx Asia/Dhaka: 2026-11-06 ~23:47
- Prompt assumed START_HERE_modmul_none.txt; exact paste unconfirmed
- Raw: exact pasted `def build_weights()` only (no shell wrapper)

## Contract (as-is)
Prompt says `np` already imported and `math` allowed; sandbox injects `np`/`math`. Missing imports are **not** an error under contract. No imports added on as-is path.

- ok=True success=True all_acc=1.0
- error_kind=None error=None
- Domain N=12544 (a,b in 1..112); shapes {'W1': [256, 226], 'W2': [113, 256]}
- Exact counts: {'correct': 12544, 'total': 12544, 'wrong': 0, 'accuracy': 1.0}
- Finite: True
- Margins: {'mean_margin_raw': 111.99997904770179, 'min_margin_raw': 111.99987773585053, 'mean_margin_norm': 10.630144151240092, 'min_margin_norm': 10.630139421616114, 'mean_prob_margin': 1.0, 'min_prob_margin': 1.0}
- Eval wall_s: 0.2824

## Diagnostic imports
Not needed (as-is path ran under contract).

## Policy
Construction success primary (t77). Formatting secondary. Same uniform extraction policy as Sonnet (no numerical repair). Sonnet: shell wrapper removed → 12544/12544 diagnostic; protocol formatting separate.
