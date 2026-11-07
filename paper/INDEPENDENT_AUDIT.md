> **v3 status note (7 Nov 2026):** This is the v2 pre-delivery audit, kept for the record. Its data checks still apply to v3. Two blockers listed below are superseded by author decisions recorded in `REVIEW_NOTES.md`: Cursor session hygiene is now an *author attestation* (still not independently verified), and CIFAR-10 is a bounded exploratory appendix with no feature-advantage claim, so the missing random-filter control is a stated limitation, not a blocker. Cursor model labels remain user-reported.

# Independent pre-delivery audit

Audit completed 7 Nov 2026, Asia/Dhaka. Verdict: **passes as an explicitly qualified review draft, not yet a submission-ready controlled comparison.** No experiments, model calls, or changes to raw measurements.

## Source checks passed
- Cursor: all 30 aggregate success flags independently resolved against individual evaluation reports, including three nested first-trial schemas. Construction count 27/30; Claude 4/5 and 5/5, Gemini 3/5 and 5/5, GPT 5/5 and 5/5. Three numerical failures are preserved.
- API: final/tables/final_claude_per_run.csv contains 105 algorithmic trials: addition 24/30, multiplication 38/45, parity 30/30 = 92/105. Multiplication hint cells 12/15,13/15,13/15 match the aggregate table.
- Local Phi/Mistral pilot: 12 B rows, 5 runnable, 7 non-runnable, zero successes. Earlier Qwen algorithmic rows independently counted in Run 1: 0/72. Different 2048 vs 8192 token limits remain explicit.
- Figure-source learning-curve copy is byte-identical to v3_run1_outputs/A_curves.csv (468 rows). Each selected optimizer/task LR has three seeds. Curves select LR by final-checkpoint mean validation accuracy, not test accuracy; bands are sample SD, not CI. Checkpoints are measured, connecting lines do not add measurements.
- Noise diagnostic verified against actual Run-1/Run-2 per_run.csv: 14 clean-successful fresh Claude constructions, one oracle, three AdamW seeds at LR 0.0003. Coordinate-wise Gaussian noise, full-domain evaluation and RNG seed 0 traced to 20_harness.py. Selection and unmatched-run caveats added; no broader robustness superiority claim.
- Both full code listings match their original raw .py sources exactly (whitespace stripped only for comparison). Embedded failure comments remain verbatim, even where their mathematical claim is wrong.
- PDF contains all 10 embedded figure images, not just files alongside the manuscript, and three numeric tables. PNG and PDF figure sets are both included. Independently rendered every page with pdftoppm; reviewed montage plus full-size data/diagram/code pages. No clipped table or text detected after corrections; code uses small but zoomable type and has original .py files alongside it.
- Public Takehi et al. page opened this turn: authors, title subject, 2 Nov 2026 date and URL match the citation. Other references retain established arXiv identifiers; this audit did not perform a full literature or novelty review.

## Material clarification made
**27/30 is mathematical construction success, not strict raw-response format compliance.** Sonnet multiplication r0 fails the strict Python-only path on a shell heredoc/self-test wrapper; a saved transparent extraction of its unchanged function succeeds on all 12,544 inputs. The earlier manuscript mentioned wrappers generally but did not attach this exception to its main number. Added explicit abstract and Methods disclosure. The construction endpoint and raw data were not changed. Do not tell readers all 30 raw responses were executed unchanged.

## Other focused corrections
- Corrected overflowing API table: readable headings and three-decimal display CIs; underlying numbers unchanged.
- Removed Figure 2 annotation/legend collision; plotted trial dots with horizontal jitter so binary values stay 0/1.
- Scoped the 30/10/60 split to algorithmic tasks; added CIFAR's actual 45k/5k training/validation and official 10k test protocol, training batch/optimizer details, LR selection, ridge details.
- Replaced stripped-text Markdown with proper tables, image links, headings and fenced verbatim code. HTML image links now relative and portable.
- Added detailed noise provenance, source hashes and reader-facing sequential figure numbering (internal filenames remain stable). Included noise source CSVs and source_protocol/ in the package.
- Fixed pandas-3 compatibility in figure-build LR grouping without changing selection semantics.
- Saved original pre-audit manuscript separately; preserved manuscript_draft/ untouched.

## Remaining scientific / publication blockers
- Cursor picker identities and fresh-Ask/empty-workspace hygiene remain user-reported/unconfirmed. Avoid provider-verified identity claims or a controlled leaderboard.
- Heterogeneous interfaces, sample sizes, budgets and extraction policies cannot establish closed-vs-open model superiority, progress over time, or general-purpose weight synthesis.
- CIFAR lacks matched random-filter controls. This remains exploratory; no feature-quality advantage established.
- Full prompt/log supplementary bundles are linked separately, not silently represented as all included in this manuscript-only zip. Human author must review provenance, venue fit, affiliation and final claims before submission.
- Main empirical evidence is capability on specified finite domains, not a novelty proof for the analytic identities or identification of recall vs reasoning.

## Paths and rebuild
Final manuscript: WSP_manuscript_v2.pdf (10 pages at audited build), .html and .md.
Package: WSP_manuscript_v2_package.zip.
Pre-audit backup: ../final_manuscript_pre_audit/.
Original user-approved draft remains ../manuscript_draft/.

Figure generation and focused layout scripts are included. Rendering requires Python with pandas/numpy/matplotlib/beautifulsoup4/markdownify and Chrome. The delivered HTML is directly printable; the full study source/log bundles remain linked in README.
