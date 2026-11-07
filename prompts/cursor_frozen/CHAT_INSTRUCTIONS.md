# Operator checklist — Cursor WSP pilot

## Before each chat
- [ ] Fresh chat (new conversation ID)
- [ ] Named model selected (not Auto); write exact string
- [ ] Reasoning / High / Medium / temperature setting written down if shown
- [ ] Date/time (Asia/Dhaka) recorded
- [ ] No repo files attached; no Agent terminal / browser / MCP tools if disableable
- [ ] System + USER prompts taken from this package only (no paraphrase)

## After the reply
- [ ] Save full raw text exactly as returned
- [ ] Note truncation / “continue” buttons / stopped-early UI state
- [ ] Do not regenerate, repair, or ask follow-ups for the recorded trial
- [ ] Fill metadata JSON from template
- [ ] Keep one folder per model slug

## Naming
`responses/<model_slug>_<task>_none_r<rep>_raw.txt`
`responses/<model_slug>_<task>_none_r<rep>_meta.json`

Example model_slug: `cursor_composer1`, `cursor_gpt54`, `cursor_sonnet45` — use the UI label you actually selected.
