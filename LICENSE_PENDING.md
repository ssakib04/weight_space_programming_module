# License: decision pending (author review required)

There is **no LICENSE file** in this release candidate, and none has been chosen. Without a license, default copyright applies, so a license must be chosen before the repository is made public.

What needs deciding:
1. **Author-written material** (paper text, figures, evaluation harness, scripts, notebooks): the author may choose any license, e.g. MIT/Apache-2.0 for code and CC BY 4.0 for the paper and figures.
2. **Model-generated responses and code** (`cursor_trials/`, `runs/*/llm_calls.jsonl`, `runs/*/generated_code/`, local-pilot outputs): these come from third-party models (Anthropic Claude via API; models accessed through Cursor; Qwen, Phi-4, and Mistral open weights). **Do not automatically relicense them under the repository's license.** Before publishing, the author should review the output and usage terms of each provider and of Cursor, and the licenses of the open-weight models (Qwen2.5, Phi-4, Mistral-Small-24B-Instruct-2501). Then either mark these directories with the applicable terms or exclude them. No model weights from these providers are included, only the small arrays that the generated code produced.
3. **Quoted third-party work** (references, the Weight-Space Programming term from Takehi et al., 2026) is cited, not redistributed.

Suggested next step: the author chooses a license for (1) and records the terms for (2) in a NOTICE file.
