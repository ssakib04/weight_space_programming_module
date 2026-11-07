# v3 revision notes

Built 7 Nov 2026 (Asia/Dhaka) from the audited v2 package (snapshot preserved at `../final_manuscript_v2_audited_snapshot_20261107T1325/`).

Changes (text and layout only; no data, score, or figure-source changes):
- Title set to the author's selection: "How Reliably Can Language Models Construct Neural Network Weights? Evidence from Modular Arithmetic Tasks" (HTML `<title>`, H1, PDF metadata, Markdown front matter).
- Abstract: reliability framed as exploratory (n=5 per cell, wide intervals, no head-to-head ranking); notes that model labels are user-reported and session conditions author-attested.
- Methods 3.3 Cursor: removed "fresh Ask not newly confirmed" and added the dated author attestation with its limits (given after collection, not independently verified).
- Limitations: attestation wording; CIFAR now an explicitly bounded exploratory appendix; sample-size wording clarified.
- CIFAR-10 moved from §4.5 to Appendix B with a no-feature-advantage statement; figures renumbered in reading order (CIFAR is now Figure 10).
- Takehi et al. reference: full title from the public page; labelled as a blog post.
- Appendix A now points to the local GitHub release candidate.

Primary numbers unchanged: Cursor 27/30 (4/5, 5/5; 3/5, 5/5; 5/5, 5/5); API modmul 38/45; API algorithmic 92/105; Phi/Mistral 0/12 (5 runnable, 7 failed); Qwen 0/72 (2048-token cap).
