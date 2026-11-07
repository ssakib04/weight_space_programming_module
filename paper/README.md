# Manuscript v3 package (pre-publication review build)

**Title:** How Reliably Can Language Models Construct Neural Network Weights? Evidence from Modular Arithmetic Tasks
**Author:** Kazi Sadman Sakib
**Status:** Review manuscript v3. Not posted, not submitted, not peer reviewed.

## Files
- `WSP_manuscript_v3.pdf` — 10 pages, A4, 10 figures, 3 tables
- `WSP_manuscript_v3.html` — editable source (relative image paths)
- `WSP_manuscript_v3.md` — Markdown companion
- `figures/` — PNG + PDF figures (unchanged from audited v2)
- `data/`, `tables/`, `source_protocol/` — figure inputs and the original pipeline source
- `SOURCE_TO_FIGURE_MANIFEST.md` — sources, hashes, reader-facing figure order
- `cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json` — dated author attestation (not independent verification)
- `REVIEW_NOTES.md`, `V3_REVISION_NOTES.md`, `INDEPENDENT_AUDIT.md` (v2 audit, still applicable to the data)
- `scripts/` — `build_figures.py` (figures from data), `make_v3.py` (v3 text revision); the v2 audit scripts are kept for the record

## Rebuild the PDF from the HTML
```bash
google-chrome --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=WSP_manuscript_v3.pdf "file://$PWD/WSP_manuscript_v3.html"
```
