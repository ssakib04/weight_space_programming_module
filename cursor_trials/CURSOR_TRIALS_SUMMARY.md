# Cursor trials summary — COMPLETE 30/30

Construction primary success = accuracy ≥ 0.99. No numerical repair. n=5 per cell — **do not rank models by reliability**.

## Success table (successes / 5)

| Model (user-reported label) | modmul/none | permadd/none |
|-----------------------------|-------------|--------------|
| Claude Sonnet 5.5 High | 4/5 | 5/5 |
| Gemini 4 Argon Medium | 3/5 | 5/5 |
| GPT-5.6 Sol Medium | 5/5 | 5/5 |

**Total construction successes: 27/30** (3 failures).

Assertion: `overall == sum(six cells) == sum(30 per-trial success flags) == 27`.
No pooled overall Wilson CI (heterogeneous model×task cells).

### Failed constructions (exact 3)
- `sonnet55high_modmul_none_r3` acc=0.9821428571428571
- `gemini4argonmedium_modmul_none_r2` acc=0.009247448979591837
- `gemini4argonmedium_modmul_none_r4` acc=0.24920280612244897

## GPT permadd per-rep (t95u)
| Rep | Valid | Success | Exact correct/acc | Connected units | Notes |
|-----|-------|---------|-------------------|-----------------|-------|
| r0 | True | True | 12769/1.0 | 8 | k=1-style 8 units |
| r1 | True | True | 12769/1.0 | 225 | Fourier 56×4 |
| r2 | True | True | 12769/1.0 | 8 | k=1-style 8 units |
| r3 | True | True | 12769/1.0 | 168 | 3 units/freq |
| r4 | True | True | 12769/1.0 | 8 | k=1-style 8 units |

## Provenance caveats
- Fresh Ask / empty workspace **not newly confirmed** for every paste.
- Model names are **user-reported** Cursor picker labels, not provider-API verified.
- t93 (empty quotes) and t94 ignored — not trials.
- Aggregate audit: an earlier undercount (three early modmul r0 successes mis-stored as null) was corrected to **27/30** without changing raw trial files.

## Artifact index
- Per-trial dirs: `cursor_trials/<slug>_<task>_none_r{0..4}/`
- Cell summaries: `*_ALL5_summary.json`
- All 30 JSON: `ALL30_per_trial.json`, `ALL30_cells_summary.json`
- Combined zip: `cursor_trials_all30_bundle.zip`
