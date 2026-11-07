# Review notes — manuscript v3 (not part of the manuscript body)

Status: pre-publication review manuscript. Not posted, not submitted. Human review is required before anything is made public.

## Resolved by author decisions (7 Nov 2026, Asia/Dhaka)
1. **Cursor session conditions.** The author attests (recorded 2026-11-07 13:20 Asia/Dhaka) that all Cursor queries used Ask mode, unchanged frozen prompts, an empty workspace, and a separate chat per query. Recorded as an **attestation**, not independent verification: `cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json`. Contemporaneous trial metadata remain unchanged. This item is no longer an open blocker, but the manuscript describes it as author-attested.
2. **CIFAR-10 scope.** The author chose to keep CIFAR-10 as an exploratory appendix without matched random-filter controls. The paper makes no feature-advantage claim, so the missing control is a stated limitation, not a submission blocker. No extra tests.
3. **Title selected:** "How Reliably Can Language Models Construct Neural Network Weights? Evidence from Modular Arithmetic Tasks". It uses a question and does not claim the approach is new.

## Still true / still disclosed
- Cursor model labels are user-reported model-picker labels, not provider-verified. Provider-side versions and sampling settings are unobservable.
- 27/30 is the primary construction endpoint; sonnet55high_modmul_none_r0 is counted after documented shell-wrapper extraction, with no numerical repair.
- Heterogeneous channels; n=5 per Cursor cell. Reliability numbers are exploratory, not a ranking.
- Qwen 0/72 used a 2048-token cap vs 8192 for Phi/Mistral; never budget-matched.
- Optional, not done: re-asking truncated API prompts at higher max_tokens; Claude API on permadd.

## Before public release
- Author decides the LICENSE (see github_release_candidate/LICENSE_PENDING.md) and reviews the redaction manifest.
- Author chooses the repository/host. No repository has been created and nothing has been published.
- Venue choice and affiliation line remain author decisions.
