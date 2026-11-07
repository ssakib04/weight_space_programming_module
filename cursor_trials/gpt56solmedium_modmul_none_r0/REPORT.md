# Cursor trial — gpt56solmedium_modmul_none_r0

## Provenance
- Model (reported): GPT-5.6 Sol Medium
- Mode: Ask instructed; **not independently confirmed** (only Sonnet was explicitly confirmed Ask)
- Time approx Asia/Dhaka: 2026-11-06 ~23:51
- Prompt assumed; exact paste unconfirmed
- Raw source: t81u inherited paste (parent transcript not readable from this evaluator session)

## Contract as-is (np/math pre-provided; no import repair; no numerical edits)
- process_reply ok=True success=True all_acc=1.0
- Domain N=12544 (a,b in 1..112)
- Shapes: {'W1': [256, 226], 'W2': [113, 256]}
- Finite: True

## Exact counts
- Torch harness (torch.float32): **12544/12544** wrong=0
- NumPy float64 forward: **12544/12544** wrong=0
- NumPy float32 forward: **12544/12544** wrong=0
- Dtype discrepancy on accuracy: **True** (same)

## Margins (do not cross-rank models on normalized margins)
- Harness: {'mean_margin_raw': 0.001573155444039374, 'min_margin_raw': 0.0015728473663330078, 'mean_margin_norm': 0.0021499738733564873, 'min_margin_norm': 0.002149552549061665, 'mean_prob_margin': 3.010481231775533e-05, 'min_prob_margin': 3.0098918598998187e-05}
- NumPy f64 min_margin_raw=0.00157318 mean_raw=0.00157318
- NumPy f32 min_margin_raw=0.00157285 mean_raw=0.00157316
- Note: Raw margins are in logit units of each network; normalized margins are within-network only and must not be used to rank models against each other.

## Capacity usage (matrix sparsity, not timed compute)
- Nonzero W1 rows: 9 indices [0, 1, 2, 3, 4, 5, 6, 7, 8]
- Nonzero W2 cols: 9 indices [0, 1, 2, 3, 4, 5, 6, 7, 8]
- Connected (nonzero W1 row ∩ nonzero W2 col): 9 → [0, 1, 2, 3, 4, 5, 6, 7, 8]
- Allocated hidden width: 256
- No claim of improved FLOPs without measurement.

## Mathematical vs formatting
- Formatting: clean `def build_weights()` (no shell wrapper) → as-is contract path OK
- Mathematical success at 0.99: **True**
