# ---------------- Configuration ----------------
import os, json, time, math, random, platform, re, traceback, gc, tempfile, subprocess, sys, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import matplotlib; import matplotlib.pyplot as plt
ON_KAGGLE = os.path.isdir("/kaggle/working")
_env = lambda k, d: os.environ.get(k, d)
QUICK = _env("WSP_QUICK", "0") == "1"   # tiny CPU smoke test
MOCK  = _env("WSP_MOCK", "0") == "1"    # replace local LLMs by a mock synthesizer (no downloads)

RUN_TASKS = {"modadd": True, "modmul": True, "parity": True, "permadd": True, "cifar": True}
RUN_A = True
RUN_LOCAL = _env("WSP_RUN_LOCAL", "1") == "1"      # local open-weight synthesizers
RUN_ORACLE_CHECK = True

# ======== CLAUDE SPEND CONTROLS (read carefully) ========
RUN_CLAUDE     = _env("WSP_RUN_CLAUDE", "0") == "1"   # master switch. False -> no Claude section at all.
DRY_RUN_CLAUDE = _env("WSP_DRY_RUN_CLAUDE", "1") == "1"  # True -> NEVER construct the Anthropic client; use cache / v2 replay / mock.
CLAUDE_BUDGET_USD = float(_env("CLAUDE_BUDGET_USD", "3.00"))    # HARD total cap for this run
CLAUDE_SECTION_CAP = {"B_algo": 2.00, "B_cifar": 0.80, "C": 0.30}   # HARD per-section caps
RUN_C_CLAUDE = _env("WSP_RUN_C", "0") == "1"                              # agentic editing with Claude (off by default)
# Cases to (re)run with Claude: (task, prompt_level, list_of_rep_indices). Reps 0-9 of v2 are already in the cache/replay,
# so fresh samples start at rep 10. Each (model, prompt, rep) is cached: a re-run never pays twice.
RUN_CLAUDE_ONLY_CASES = [("modmul", "full", list(range(10, 15))), ("modmul", "partial", list(range(10, 15))),
                         ("modmul", "none", list(range(10, 15))), ("cifar", "full", [10, 11]), ("modadd", "none", [])]
CLAUDE_MODEL_ID    = _env("CLAUDE_MODEL_ID", "claude-sonnet-5-5")
CLAUDE_TEMPERATURE = 0.0   # retried without temperature if the model rejects it
CLAUDE_MAX_TOKENS  = {"algo": 6000, "cifar": 12000, "C": 6000}
PRICE_PER_MTOK     = {"input": 2.0, "output": 10.0}
CACHE_DIR = _env("WSP_CACHE_DIR", "/kaggle/working/llm_cache" if ON_KAGGLE else "./llm_cache")
# Optional: v2 llm_calls.jsonl (e.g. attached as a Kaggle dataset) to seed the cache and replay offline.
V2_LLM_LOG = _env("WSP_V2_LLM_LOG", next((p for p in ["/kaggle/input/wsp-v2-outputs/llm_calls.jsonl", "./v2_outputs/llm_calls.jsonl"]
                                          if os.path.exists(p)), ""))

# Local models (one loaded at a time). Rows are labelled by the model id that ACTUALLY loaded.
SYNTHESIZERS = [
    {"name": "qwen2.5-coder-14b", "kind": "hf", "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct"},
    {"name": "qwen2.5-7b", "kind": "hf", "model_id": "Qwen/Qwen2.5-7B-Instruct"},
    {"name": "llama-3.1-8b", "kind": "hf", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "requires_hf_token": True},
]
if MOCK: SYNTHESIZERS = [{"name": "mock", "kind": "mock", "model_id": "mock"}]
LOCAL_MAX_NEW_TOKENS = 2048
LOCAL_DETERMINISTIC = False          # True -> greedy (do_sample=False): repeats collapse to 1 per cell
LOCAL_TEMPERATURE, LOCAL_TOP_P, LOCAL_SEED_BASE = 0.7, 0.95, 1000   # sampling is seeded per repeat (seed = base + rep)

PROMPT_LEVELS = ["full", "partial", "none"]
N_REPEATS_LOCAL = 3
N_REPEATS_CIFAR_LOCAL = 2
SUCCESS_THRESH = 0.99      # algorithmic tasks only. CIFAR reports accuracy only.
CIFAR_CHANCE, CIFAR_ABOVE_CHANCE_PTS = 0.10, 0.05

SEEDS_A = [0, 1, 2]
LR_GRID = {"adamw": [3e-4, 1e-3, 3e-3], "sgd": [0.03, 0.1, 0.3]}
WEIGHT_DECAY = {"adamw": 1.0, "sgd": 1e-4}
CHECKPOINTS = [100, 500, 1000, 3000, 10000, 20000]
SPLIT = (0.3, 0.1)
CIFAR_STEPS = [200, 1000, 3000]
CIFAR_LR_GRID = {"adam": [3e-4, 1e-3], "sgd": [0.01, 0.05]}
CIFAR_BATCH = 128
C_MAX_TARGETS_PER_TASK, C_ITERS = 2, 2
TIME_LIMIT = {"A_algo": 75*60, "A_cifar": 25*60, "B_claude": 60*60, "B_hf_each": 70*60, "C": 20*60}
SANDBOX_TIMEOUT = {"algo": 120, "cifar": 300}

if QUICK:
    SEEDS_A = [0]; LR_GRID = {"adamw": [1e-3], "sgd": [0.1]}; CHECKPOINTS = [20, 50]
    CIFAR_STEPS = [5, 10]; CIFAR_LR_GRID = {"adam": [1e-3]}
    N_REPEATS_LOCAL = 2; N_REPEATS_CIFAR_LOCAL = 1; C_ITERS = 1

OUT_DIR = "/kaggle/working" if ON_KAGGLE else os.path.abspath(_env("WSP_OUT", "./wsp_outputs_v3"))
GEN_DIR = f"{OUT_DIR}/generated_code"; os.makedirs(GEN_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_START = time.time()
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
print("device", DEVICE, "| torch", torch.__version__, "| out", OUT_DIR, "| QUICK", QUICK, "| MOCK", MOCK,
      "| RUN_CLAUDE", RUN_CLAUDE, "| DRY_RUN_CLAUDE", DRY_RUN_CLAUDE, "| budget $%.2f" % CLAUDE_BUDGET_USD)
