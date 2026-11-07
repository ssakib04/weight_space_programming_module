# Protocol freeze — 2026-11-07T00:04 Asia/Dhaka

Declared **after** the first three modmul/none r0 Cursor trials (Sonnet, Gemini, GPT). This is a **post-pilot design freeze**, not a preregistration before seeing data.

Frozen choices:
- Models: Claude Sonnet 5.5 High; Gemini 4 Argon Medium; GPT-5.6 Sol Medium (exact UI labels at send time)
- Tasks: modmul/none and permadd/none only
- n=5 independent fresh Ask chats per model × task (30 total)
- Prompts: START_HERE_modmul_none.txt and START_HERE_permadd_none.txt (byte-stable; hashes in prompts_start_here_manifest.json)
- Channel: Cursor Ask; report separately from API / Kaggle local
- Primary endpoint: construction success (accuracy ≥ 0.99 on full task domain)
- Formatting secondary; uniform extraction; no numerical repair of weights
- permadd uses fixed VAL/PERM instance — reliability on one mapping, not recall-proof / novel mapping
- Insufficient n for ranking model reliability with certainty
