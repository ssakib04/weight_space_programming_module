"""Exercise call_hf (chat template, seeded sampling, greedy determinism, logging, labels) with Qwen2.5-0.5B-Instruct on CPU.
bitsandbytes 4-bit loading itself cannot be tested on CPU: load_hf is replaced by an fp32 load of the tiny model."""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__)); from common import load
ns = load()
from transformers import AutoTokenizer, AutoModelForCausalLM
mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid); mdl = AutoModelForCausalLM.from_pretrained(mid).eval()
ns["_HF"].update(tok=tok, model=mdl, id=mid); ns["LOCAL_MAX_NEW_TOKENS"] = 24
syn = {"name": "tiny", "kind": "hf", "model_id": mid, "resolved_id": mid}
ok = True
ns["LOCAL_DETERMINISTIC"] = True
a = ns["call_hf"](syn, "sys", "hello", "t", rep=0)[0]; b = ns["call_hf"](syn, "sys", "hello", "t", rep=5)[0]
print("greedy identical across reps:", a == b); ok &= a == b
ns["LOCAL_DETERMINISTIC"] = False
c = ns["call_hf"](syn, "sys", "hello", "t", rep=3)[0]; d = ns["call_hf"](syn, "sys", "hello", "t", rep=3)[0]
print("seeded sampling reproducible for same rep:", c == d); ok &= c == d
row = ns["synth_once"](syn, "modadd", "full", 0)
print("synth_once row label:", row["synth"], "| requested:", row["synth_requested"], "| error_kind:", row["error_kind"], "| stop:", row.get("stop_reason"))
ok &= row["synth"] == mid and row["error_kind"] in ("no_code", "truncated", "syntax")
# gated model without HF_TOKEN -> explicit skip, no fallback
os.environ.pop("HF_TOKEN", None)
try: ns["load_hf"]({"name": "llama", "kind": "hf", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "requires_hf_token": True}); ok = False
except RuntimeError as e: print("gated skip:", str(e)[:90]); ok &= "SKIPPED" in str(e)
last = json.loads(open(ns["LLM_LOG"]).readlines()[-1]); print("log has seed/do_sample:", "seed" in last and "do_sample" in last)
print("HF PATH TEST:", "PASS" if ok else "FAIL")
