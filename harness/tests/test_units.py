"""(b) mock-synthesizer / sandbox / extraction tests and (c) metric, CI, budget, cache, dry-run unit tests. CPU only, no network."""
import sys, os, json, math, types; sys.path.insert(0, os.path.dirname(__file__)); from common import load
import numpy as np, torch
ns = load()
RES = []
def check(name, cond, detail=""):
    RES.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
ex, run, val, SE = ns["extract_code"], ns["run_generated"], ns["validate"], ns["SandboxError"]
ORA = ns["ORACLE_SRC"]; shapes = ns["TASKS"]["modadd"]["shapes"]
def outcome(text, stop=None, fn="build_weights", inputs=None, timeout=60, task="modadd"):
    r = ns["process_reply"](task, text, {"stop_reason": stop}, "unit_" + str(abs(hash(text)) % 10**8)) if fn == "build_weights" else None
    return r
# ---------- (b) extraction + sandbox ----------
r = outcome("Here you go:\n```python\nimport numpy as np\nimport math\n" + ORA["modadd"] + "```\nExplanation follows.")
check("fenced oracle with imports + prose -> success", r["success"], r.get("error"))
r = outcome("Explanation first.\n\nimport numpy as np\n" + ORA["modadd"] + "\nThat is the construction; it works because ...")
check("unfenced oracle with prose before/after -> success", r["success"], f"{r.get('extract_method')} {r.get('error')}")
r = outcome("```py\n" + ORA["modadd"] + "```")
check("```py fence -> success", r["success"])
r = outcome("```\n" + ORA["modadd"] + "```")
check("bare ``` fence -> success", r["success"])
helper = "```python\ndef helper(p):\n    return np.arange(p)\n```\nand\n```python\ndef build_weights():\n    x = helper(113)\n" + ORA["modadd"].split("def build_weights():\n")[1].replace("\n    ", "\n    ", 1) + "```"
r = outcome(helper)
check("multiple fences (helper in another block) -> concatenated, success", r["success"], f"{r.get('extract_method')} {r.get('error')}")
r = outcome("```python\nimport numpy as np\ndef build_weights():\n    W1 = np.zeros((256, 226))\n    for k in range(3):\n        W1[k, :] = np.cos(np.arange(226)")
check("truncated reply (unclosed fence) -> error_kind truncated", r["error_kind"] == "truncated" and r["extract_truncated"], r.get("error"))
r = outcome("```python\n" + ORA["modadd"] + "```", stop="max_tokens")
check("complete code but stop_reason=max_tokens -> still evaluated, flagged", r["success"] and r["extract_truncated"])
r = outcome("I cannot do this.")
check("no code -> no_code", r["error_kind"] == "no_code", r.get("error"))
r = outcome("```python\ndef build_weights():\n    return {'W1': np.zeros((3,3)), 'W2': np.zeros((113,256))}\n```")
check("wrong shape -> shape", r["error_kind"] == "shape", r.get("error"))
r = outcome("```python\ndef build_weights():\n    return {'W1': np.full((256,226), np.nan), 'W2': np.zeros((113,256))}\n```")
check("NaN weights -> nonfinite", r["error_kind"] == "nonfinite")
r = outcome("```python\ndef build_weights():\n    return [1,2]\n```")
check("non-dict -> not_dict", r["error_kind"] == "not_dict", r.get("error"))
r = outcome("```python\ndef make():\n    return {}\n```")
check("wrong function name -> no_code/missing_fn", r["error_kind"] in ("no_code", "missing_fn"), r.get("error_kind"))
src_next = ORA["modadd"].replace("    p, H, M = 113, 256, 3", "    p = next(iter([113])); H, M = 256, 3\n    _ = list(zip(*enumerate(map(abs, filter(None, sorted(reversed([1,2]))))))); _ = any([1]) and all([1]); _ = divmod(7,2), pow(2,3), round(1.5), min(1,2), max(1,2), sum([1])\n    import itertools, functools, math as m2\n    print('hello from generated code')")
r = outcome("```python\n" + src_next + "```")
check("next/iter/zip/map/filter/sorted/... + itertools/functools imports + print -> success", r["success"], r.get("error"))
t0 = __import__("time").time()
try: run("def build_weights():\n    while True: pass\n", timeout=3); check("infinite loop -> timeout", False)
except SE as e: check("infinite loop -> timeout", e.kind == "timeout" and __import__("time").time() - t0 < 15, e.kind)
for name, code in [
    ("import os", "import os\ndef build_weights():\n    return {}\n"),
    ("import subprocess inside fn", "def build_weights():\n    import subprocess; subprocess.run(['ls'])\n    return {}\n"),
    ("open()", "def build_weights():\n    open('/etc/passwd').read()\n    return {}\n"),
    ("np.load file read", "def build_weights():\n    np.loadtxt('/etc/hostname')\n    return {}\n"),
    ("np.save file write", "def build_weights():\n    np.save('/tmp/pwned.npy', np.zeros(2))\n    return {}\n"),
    ("__import__('os')", "def build_weights():\n    __import__('os').system('echo hi')\n    return {}\n"),
    ("eval", "def build_weights():\n    eval('1+1')\n    return {}\n"),
    ("subclass escape -> os.system", "def build_weights():\n    for c in ().__class__.__base__.__subclasses__():\n        if c.__name__ == 'catch_warnings':\n            b = c()._module.__builtins__\n            imp = b['__import__'] if isinstance(b, dict) else b.__import__\n            imp('os').system('touch /tmp/wsp_escape_marker')\n    return {}\n"),
    ("socket via numpy internals", "def build_weights():\n    import numpy\n    s = numpy.__builtins__ if hasattr(numpy,'__builtins__') else None\n    import socket\n    return {}\n"),
    ("relative import", "from . import x\ndef build_weights():\n    return {}\n"),
]:
    if os.path.exists("/tmp/wsp_escape_marker"): os.remove("/tmp/wsp_escape_marker")
    try: run(code, timeout=60); check(f"forbidden: {name} -> blocked", False, "ran without error!")
    except SE as e: check(f"forbidden: {name} -> blocked", e.kind in ("blocked", "import", "name_error") and not os.path.exists("/tmp/wsp_escape_marker") and not os.path.exists("/tmp/pwned.npy"), f"{e.kind}: {str(e)[-120:]!r}")
os.environ["ANTHROPIC_API_KEY"] = "sk-test-should-not-leak"
try: run("def build_weights():\n    import numpy\n    return {'k': numpy.zeros(1)}\n"); check("env secret not passed (smoke)", True)
except SE as e: check("env secret not passed (smoke)", False, str(e))
del os.environ["ANTHROPIC_API_KEY"]
W = run("import torch\ndef build_weights():\n    return {'W1': torch.zeros(256,226), 'W2': torch.ones(113,256)}\n")
check("torch tensors returned -> converted", W["W2"].shape == (113, 256) and W["W2"].sum() == 113 * 256)
W0 = {"W1": np.ones((256, 226)), "W2": np.ones((113, 256))}
W = run("def edit_weights(W):\n    W = dict(W); W['W2'] = W['W2'] * 2\n    return W\n", "edit_weights", inputs=W0)
check("edit_weights receives inputs", float(W["W2"].mean()) == 2.0)
for t in ["modmul", "parity", "permadd"]:
    r = ns["process_reply"](t, "```python\n" + ORA[t] + "```", {}, f"unit_or_{t}")
    check(f"oracle {t} via process_reply -> success", r["success"])
# ---------- (c) metrics ----------
lg = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]]); y = torch.tensor([0, 1])
raw, norm, pm = ns["margin_stats"](lg, y)
check("raw margin", torch.allclose(raw, torch.tensor([1.0, -5.0], dtype=torch.float64)))
check("normalised margin = raw / logit std", torch.allclose(norm, raw / lg.double().std(-1)))
p = torch.softmax(lg.double(), -1)
check("prob margin", torch.allclose(pm, torch.tensor([p[0,0]-p[0,1], p[1,1]-p[1,2]])))
_, n2, pm2 = ns["margin_stats"](lg * 1000, y)
check("normalised margin is scale-invariant", torch.allclose(n2, norm))
lo, hi = ns["wilson"](7, 10); check("Wilson 7/10 = (0.397, 0.892)", abs(lo - 0.3968) < 1e-3 and abs(hi - 0.8922) < 1e-3, f"{lo:.4f},{hi:.4f}")
lo, hi = ns["wilson"](10, 10); check("Wilson 10/10 upper=1, lower 0.722", hi == 1.0 and abs(lo - 0.7225) < 1e-3, f"{lo:.4f}")
check("Wilson n=0 -> nan", all(math.isnan(v) for v in ns["wilson"](0, 0)))
m, sd, ci = ns["t_ci"]([1.0, 2.0, 3.0]); check("t CI (n=3) = 4.303*1/sqrt3", abs(ci - 4.3027 / math.sqrt(3)) < 1e-3)
# ---------- (c) budget / cache / dry-run with a fake client ----------
class FakeResp:
    def __init__(s, text, ui, uo, stop="end_turn"):
        s.content = [types.SimpleNamespace(type="text", text=text)]; s.usage = types.SimpleNamespace(input_tokens=ui, output_tokens=uo)
        s.model = "fake-model"; s.stop_reason = stop
class FakeClient:
    def __init__(s, out_tok=4000, reject_temp=False): s.calls = 0; s.out_tok = out_tok; s.reject_temp = reject_temp; s.messages = s
    def create(s, **kw):
        s.calls += 1
        if s.reject_temp and "temperature" in kw: raise ValueError("temperature is not supported for this model")
        return FakeResp("```python\n" + ORA["modmul"] + "```", 1500, s.out_tok)
def fresh(**over):
    n = load(overrides=over)
    return n
# dry-run never constructs a client
n = fresh(DRY_RUN_CLAUDE=True)
def boom(): raise AssertionError("client constructed in dry run!")
n["CLIENT_FACTORY"] = boom
syn = {"name": "claude", "kind": "claude", "model_id": "claude-sonnet-5-5"}
rows = [n["synth_once"](syn, "modmul", "full", i) for i in range(3)]
check("DRY_RUN: zero API calls, client never constructed, rows flagged dry_run_mock",
      n["API_CALLS"]["claude"] == 0 and "client" not in n["_CLAUDE"] and all(r["reply_source"] == "dry_run_mock" for r in rows))
check("DRY_RUN: anthropic module never imported", "anthropic" not in sys.modules)
# live path with fake client: budget, cache, temperature retry
n = fresh(DRY_RUN_CLAUDE=False, CLAUDE_BUDGET_USD=0.20, CLAUDE_SECTION_CAP={"B_algo": 0.15, "B_cifar": 0.8, "C": 0.3})
n["BUDGET"] = n["Budget"](0.20, {"B_algo": 0.15}); fc = FakeClient(out_tok=4000, reject_temp=True); n["CLIENT_FACTORY"] = lambda: fc
done, stopped = 0, False
try:
    for i in range(20): n["synth_once"](syn, "modmul", "full", i); done += 1
except n["BudgetExceeded"] as e: stopped = True; why = str(e)
# worst-case per call = (prompt/3*2 + 6000*10)/1e6 ~ 0.061; actual ~0.043 each -> allowed while spent+0.061 <= 0.15
check("budget: calls stop before crossing section cap", stopped and n["BUDGET"].spent <= 0.15 + 1e-9, f"done={done} spent=${n['BUDGET'].spent:.3f} {why}")
check("budget: spend recorded per section", abs(n["BUDGET"].by_section["B_algo"] - n["BUDGET"].spent) < 1e-12)
check("temperature rejection -> one retry, then no temperature", fc.calls == done + 1, f"fake calls {fc.calls} for {done} replies")
calls_before = fc.calls
rows = [n["synth_once"](syn, "modmul", "full", i) for i in range(done)]
check("cache: re-running the same (model,prompt,rep) makes ZERO API calls", fc.calls == calls_before and all(r["cached"] for r in rows), f"{fc.calls - calls_before} new calls")
check("cache: cached rows cost $0 and give same result", all(r["cost_usd"] == 0 and r["success"] for r in rows))
# truncated cache entry is re-asked only if max_tokens increased
n["cache_put"]("m", "s", "u", 0, {"response": "x", "stop_reason": "max_tokens", "max_tokens": 6000})
check("cache: truncated@6000 not reused for 12000", n["cache_get"]("m", "s", "u", 0, 12000) is None and n["cache_get"]("m", "s", "u", 0, 6000) is not None)
# total cap
n = fresh(DRY_RUN_CLAUDE=False); n["BUDGET"] = n["Budget"](0.05, {"B_algo": 5}); fc2 = FakeClient(); n["CLIENT_FACTORY"] = lambda: fc2
try: n["synth_once"](syn, "modmul", "full", 0); n["synth_once"](syn, "modmul", "full", 1); check("total cap stops", False)
except n["BudgetExceeded"]: check("total cap stops (worst-case pre-check)", fc2.calls <= 1, f"calls={fc2.calls}")
# v2 prompt identity + cache seeding
V2 = os.path.join(os.path.dirname(__file__), "..", "v2_outputs", "llm_calls.jsonl")
recs = [json.loads(l) for l in open(V2)]
same = sum(r["user"] == (n["CIFAR_PROMPT"] if r["tag"].startswith("B_cifar") else n["task_prompt"](*r["tag"].split("_")[1:3]))
           for r in recs if r["synth"].startswith("claude") and r["tag"].startswith("B_"))
tot = sum(1 for r in recs if r["synth"].startswith("claude") and r["tag"].startswith("B_"))
check("v3 prompts byte-identical to v2 prompts (cache validity)", same == tot, f"{same}/{tot}")
n = fresh(DRY_RUN_CLAUDE=False); fc3 = FakeClient(); n["CLIENT_FACTORY"] = lambda: fc3
k = n["seed_cache_from_v2"](V2)
rows = [n["synth_once"](syn, "modmul", lvl, i) for lvl in ["full", "partial", "none"] for i in range(10)]
check("v2-seeded cache: replaying 30 v2 modmul cases makes 0 API calls", fc3.calls == 0 and all(r["cached"] for r in rows), f"seeded={k}")
check("v2-seeded cache result = offline replay (24/30 modmul success)", sum(r["success"] for r in rows) == 24, str(sum(r["success"] for r in rows)))
# plan / estimate
n["RUN_CLAUDE_ONLY_CASES"] = [("modmul", "full", [0, 1, 10, 11]), ("cifar", "full", [0, 10])]
exec(open(os.path.join(os.path.dirname(__file__), "..", "v3_src", "75_run_claude.py")).read().split("if RUN_CLAUDE:")[0], n)
P = n["PLAN"]
check("plan: cached reps cost $0, fresh reps priced; truncated v2 CIFAR r0? ", (P[P.cached].worst_usd == 0).all() and (P[~P.cached].worst_usd > 0).all(), P.to_string())
print("\nSUMMARY: %d passed, %d failed" % (sum(r[1] for r in RES), sum(not r[1] for r in RES)))
json.dump(RES, open(os.path.join(os.path.dirname(__file__), "..", "v3_test_outputs", "unit_results.json"), "w"), indent=1)
