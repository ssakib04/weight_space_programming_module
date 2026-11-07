# Cursor WSP run order (30 total)

**Reuse prompts?** Yes — same files every time. Do **not** rewrite them.
- modmul: `START_HERE_modmul_none.txt` (unchanged)
- permadd: `START_HERE_permadd_none.txt` (new; same delimiter scheme)

**Design:** 3 named models × 2 tasks × 5 independent fresh Ask chats = **30** responses.

| Model | modmul/none | permadd/none |
|-------|-------------|--------------|
| Claude Sonnet 5.5 High | r0 **done** → need r1–r4 (**4**) | r0–r4 (**5**) |
| Gemini 4 Argon Medium | r0 **done** → need r1–r4 (**4**) | r0–r4 (**5**) |
| GPT-5.6 Sol Medium | r0 **done** → need r1–r4 (**4**) | r0–r4 (**5**) |

**Remaining: 27** (12 more modmul + 15 permadd). Do not renumber existing `*_r0` artifacts.

## Recommended next 3 (then the other 24)
1. permadd r0 — Claude Sonnet 5.5 High  
2. permadd r0 — Gemini 4 Argon Medium  
3. permadd r0 — GPT-5.6 Sol Medium  
Then finish remaining modmul r1–r4 and permadd r1–r4 per model (24 left after those 3).

## Each trial
- Fresh Ask chat; named model (not Auto); identical settings; paste START_HERE file **unchanged**
- Save **entire** raw assistant reply outside the empty workspace
- No execute / repair / regenerate
- Folder naming: `cursor_trials/<modelslug>_<task>_none_r<rep>/`

Fixed permadd VAL mapping repeats = reliability on **one** instance, not a novel mapping / recall-proof claim.
