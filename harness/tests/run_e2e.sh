#!/bin/bash
# (d) end-to-end execution of the generated notebook on CPU: QUICK steps, mock local synthesizer, Claude section in DRY-RUN
# (cache seeded from v2 log + mock), real CIFAR-10 subset via HF parquet fallback. No API key present -> any real call would fail loudly.
set -e; cd "$(dirname "$0")/.."; source .venv/bin/activate
OUT=${1:-/tmp/wsp_e2e}; rm -rf "$OUT"; mkdir -p "$OUT"
env -u ANTHROPIC_API_KEY WSP_QUICK=1 WSP_MOCK=1 WSP_RUN_CLAUDE=1 WSP_DRY_RUN_CLAUDE=1 WSP_RUN_C=1 WSP_OUT="$OUT" WSP_CACHE_DIR="$OUT/llm_cache" \
  python - <<'PY'
import nbformat, os, time
from nbclient import NotebookClient
nb = nbformat.read("wsp_modular_addition_v3.ipynb", as_version=4); t = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbformat.write(nb, os.path.join(os.environ["WSP_OUT"], "executed_v3.ipynb")); print("executed in %.0fs" % (time.time() - t))
PY
