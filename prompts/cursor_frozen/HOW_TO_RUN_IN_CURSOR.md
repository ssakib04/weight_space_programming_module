# How to run the modmul/none WSP prompt in Cursor (pilot)

Official references:
- Ask mode (read-only; can still explore/search): https://cursor.com/help/ai-features/ask-mode
- Agent mode (can edit files / run tools): https://cursor.com/help/ai-features/agent
- Rules (project vs user; apply across modes): https://cursor.com/docs/rules

## Goal for this pilot
One fresh chat → one named model → paste `START_HERE_modmul_none.txt` unchanged → save the entire assistant reply. No regenerating, fixing, or executing.

## Recommended workspace
Open Cursor on an **empty folder** that is **not** your coding project and has **no** `.cursor/rules` / `AGENTS.md`. That avoids automatic project rules and open-file context from your usual repo. **User rules** (global) and Cursor’s own system/tooling context may still apply; we cannot promise a fully clean API-equivalent channel.

## Steps
1. File → Open Folder → empty directory (or new empty folder).
2. Open Agent panel: Ctrl+I (Windows/Linux) or Cmd+I (macOS) per Cursor help.
3. Mode: prefer **Ask** (read-only; will not edit files). Ask can still search/read if files exist—keep the folder empty. Do **not** claim all tools are disabled.
4. Model picker: choose a **named** model (from your earlier screenshot examples: Claude Sonnet 5.5 High, Gemini 3 Pro or Gemini 4 Argon Medium, GPT-5.x Sol Medium—use the **exact label shown now**). Do **not** use Auto.
5. Start a **new chat** for this trial (use the UI control that creates a new conversation; if unsure, screenshot the panel rather than guessing a shortcut).
6. Copy **everything** in `START_HERE_modmul_none.txt` and paste as the **only** message. Send unchanged—no “for a paper”, no @files.
7. Wait for one complete reply. Do not click regenerate / continue-as-repair / “fix this” / run code.
8. Copy the **entire** assistant message (not only the code fence) into a file **outside** this empty folder, e.g. Desktop/`wsp_cursor_responses/`.
9. Record: exact model label, High/Medium if shown, Asia/Dhaka date/time, Ask vs Agent, whether any tools visibly ran, truncation/stop.
10. Close the chat before the next model/trial.

## One message vs API
Cursor chat usually has a single user text box. The START_HERE file therefore embeds SYSTEM then USER with explicit delimiters. A direct API call would set system and user as separate fields without those delimiter lines.
