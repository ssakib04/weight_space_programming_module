# ---------------- Sandboxed execution, robust code extraction, evaluation, metrics ----------------
# Fault-isolation harness (separate process + timeout + whitelist imports + audit hook). It is NOT a hardened security sandbox;
# all generated code is logged for review.
RUNNER = r'''
import sys, json, math, builtins as _b
import numpy as np
src_path, out_path, fn_name, in_path, want_torch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
src = open(src_path).read()
INPUTS = dict(np.load(in_path))
import itertools, functools, collections, operator, cmath, fractions, random, typing, string, statistics, numbers, dataclasses, copy, heapq, bisect
import numpy.linalg, numpy.fft, numpy.random, numpy.polynomial, numpy.lib.stride_tricks
if want_torch:
    import torch, torch.nn, torch.nn.functional
ALLOWED_MODULES = {"numpy", "math", "itertools", "functools", "collections", "operator", "cmath", "fractions", "random", "typing",
                   "string", "statistics", "numbers", "dataclasses", "copy", "heapq", "bisect", "torch", "scipy"}
_real_import = _b.__import__
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is blocked in the sandbox")
    if root == "torch" and "torch" not in sys.modules:
        raise ImportError("torch not preloaded (pass want_torch)")
    return _real_import(name, globals, locals, fromlist, level)
SAFE_NAMES = ["abs","all","any","ascii","bin","bool","bytearray","bytes","callable","chr","complex","dict","divmod","enumerate",
  "filter","float","format","frozenset","hasattr","hash","hex","id","int","isinstance","issubclass","iter","len","list","map","max",
  "min","next","object","oct","ord","pow","print","range","repr","reversed","round","set","slice","sorted","str","sum","super",
  "tuple","type","zip","getattr","setattr","property","staticmethod","classmethod","NotImplemented","Ellipsis","None","True","False",
  "Exception","ValueError","TypeError","KeyError","IndexError","AssertionError","ArithmeticError","ZeroDivisionError",
  "RuntimeError","StopIteration","NotImplementedError","AttributeError","OverflowError","LookupError","ImportError","__build_class__"]
safe = {k: getattr(_b, k) for k in SAFE_NAMES if hasattr(_b, k)}
safe["__import__"] = _safe_import
def _blocked(name):
    def f(*a, **k): raise PermissionError(f"sandbox: builtin {name}() is blocked")
    return f
for _n in ["open", "eval", "exec", "compile", "input", "breakpoint", "globals", "locals", "vars", "memoryview", "exit", "quit", "help"]:
    safe[_n] = _blocked(_n)
_ALLOW = [False]
BLOCK_PREFIX = ("subprocess.", "os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn", "os.kill", "os.remove", "os.rmdir",
                "os.rename", "os.unlink", "os.truncate", "os.chmod", "os.chown", "shutil.", "socket.", "urllib.", "http.", "ctypes.",
                "pty.", "fcntl.", "os.putenv", "os.chdir", "code.__new__")
def _hook(ev, args):
    if _ALLOW[0]: return
    if ev == "open":
        path, mode = str(args[0]), str(args[1] or "r")
        if path.endswith((".py", ".pyc", ".so")) and path.startswith(_LIB_ROOTS) and not any(c in mode for c in "wax+"):
            return                                   # lazy imports of stdlib / site-packages modules (read-only)
        raise PermissionError(f"sandbox: file access blocked ({args[0]!r})")
    if ev.startswith(BLOCK_PREFIX):
        raise PermissionError(f"sandbox: blocked operation {ev}")
_STAGE = [0]
import sysconfig as _sc
_LIB_ROOTS = tuple({sys.prefix, sys.base_prefix, sys.exec_prefix, _sc.get_paths()["stdlib"], _sc.get_paths()["purelib"], _sc.get_paths()["platlib"]})
ns = {"np": np, "numpy": np, "math": math, "__builtins__": safe, "__name__": "generated"}
if want_torch: ns.update(torch=torch, nn=torch.nn, F=torch.nn.functional)
code_obj = compile(src, "<generated>", "exec")
sys.addaudithook(_hook)
exec(code_obj, ns)           # audit 'exec' allowed only for this top-level call
_STAGE[0] = 1
if fn_name not in ns or not callable(ns[fn_name]):
    names = [k for k, v in ns.items() if callable(v) and not k.startswith("_") and k not in ("np", "numpy", "math", "torch", "nn", "F")]
    print("SANDBOX_MISSING_FN " + json.dumps(names), file=sys.stderr); sys.exit(3)
W = ns[fn_name]() if fn_name == "build_weights" else ns[fn_name](dict(INPUTS))
def _arr(v):
    if hasattr(v, "detach"): v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)
if not isinstance(W, dict):
    print("SANDBOX_NOT_DICT " + type(W).__name__, file=sys.stderr); sys.exit(4)
arrs = {str(k): _arr(v) for k, v in W.items()}
_ALLOW[0] = True
np.savez(out_path, **arrs)
'''

class SandboxError(RuntimeError):
    def __init__(self, kind, msg): super().__init__(f"[{kind}] {msg}"); self.kind = kind

FENCE_RE = re.compile(r"(?:^|\n)[ \t]*(```|~~~)[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:\n[ \t]*\1[ \t]*(?=\n|$)|\Z)", re.S)

def _compiles(s):
    try: compile(s, "<x>", "exec"); return True
    except SyntaxError: return False

def extract_code(text, fn_name="build_weights", stop_reason=None):
    """Return (code, info). Handles ```python/```py/``` fences, multiple fences, leading/trailing prose, unclosed (truncated) fences.
    Never strips imports. info = {method, n_fences, truncated, compiles}."""
    text = (text or "").replace("\r\n", "\n")
    info = {"n_fences": 0, "truncated": stop_reason == "max_tokens", "method": None, "compiles": False}
    blocks = []
    for m in FENCE_RE.finditer(text):
        lang, body = m.group(2).lower(), m.group(3)
        closed = bool(re.search(r"\n[ \t]*" + re.escape(m.group(1)) + r"[ \t]*$", m.group(0)))
        if lang in ("", "python", "py", "python3", "numpy"): blocks.append((body, closed))
    info["n_fences"] = len(blocks)
    if any(not c for _, c in blocks): info["truncated"] = True
    cands = []
    if blocks:
        with_fn = [b for b, _ in blocks if re.search(rf"def\s+{fn_name}\s*\(", b)]
        # several blocks: concatenate (helpers may live in other blocks; a later re-definition wins, i.e. the final version)
        if len(blocks) > 1: cands.append(("fences_concat", "\n\n".join(b for b, _ in blocks)))
        if with_fn: cands.append(("fence_with_fn", with_fn[-1]))
        cands.append(("longest_fence", max((b for b, _ in blocks), key=len)))
    else:
        lines = text.split("\n")
        start = next((i for i, l in enumerate(lines) if re.match(r"\s*(import |from \w|def |@|[A-Za-z_]\w*\s*=)", l)), None)
        if start is not None:
            body = lines[start:]
            for end in range(len(body), 0, -1):          # trim trailing prose until it compiles
                s = "\n".join(body[:end])
                if _compiles(s): cands.append(("unfenced", s)); break
            else: cands.append(("unfenced_raw", "\n".join(body)))
        else: cands.append(("raw", text))
    for method, s in cands:
        if _compiles(s) and re.search(rf"def\s+{fn_name}\s*\(", s):
            info.update(method=method, compiles=True); return s, info
    method, s = cands[0]
    info.update(method=method, compiles=_compiles(s)); return s, info

def classify_error(msg):
    m = str(msg)
    for kind, pats in [("truncated", ["[truncated]"]), ("timeout", ["[timeout]", "TimeoutExpired"]),
                       ("blocked", ["sandbox: ", "is blocked in the sandbox"]), ("syntax", ["SyntaxError", "IndentationError"]),
                       ("missing_fn", ["SANDBOX_MISSING_FN", "[missing_fn]"]), ("not_dict", ["SANDBOX_NOT_DICT", "weights must be a dict"]),
                       ("shape", ["shape", "missing parameter"]), ("nonfinite", ["non-finite"]), ("name_error", ["NameError"]),
                       ("import", ["ImportError", "ModuleNotFoundError"]), ("no_code", ["[no_code]"])]:
        if any(p in m for p in pats): return kind
    return "runtime"

def run_generated(code_str, fn_name="build_weights", inputs=None, timeout=120, want_torch=None):
    """Run generated code in a SUBPROCESS. Returns dict of float64 arrays. Raises SandboxError(kind, ...)."""
    if want_torch is None: want_torch = bool(re.search(r"\btorch\b", code_str))
    if not _compiles(code_str):
        try: compile(code_str, "<generated>", "exec")
        except SyntaxError as e: raise SandboxError("syntax", f"SyntaxError: {e}")
    d = tempfile.mkdtemp(); src, out, inp, run = f"{d}/g.py", f"{d}/w.npz", f"{d}/in.npz", f"{d}/runner.py"
    open(src, "w").write(code_str); open(run, "w").write(RUNNER)
    np.savez(inp, **(inputs or {"_": np.zeros(1)}))
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "HF_TOKEN", "KAGGLE_KEY")}
    env.update(OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    try:
        r = subprocess.run([sys.executable, "-I", run, src, out, fn_name, inp, "1" if want_torch else "0"],
                           capture_output=True, text=True, timeout=timeout, cwd=d, env=env)
    except subprocess.TimeoutExpired:
        raise SandboxError("timeout", f"generated code exceeded {timeout}s")
    if r.returncode != 0:
        err = r.stderr[-800:]
        raise SandboxError(classify_error(err), "generated code failed: " + err)
    return {k: v for k, v in np.load(out).items()}

def validate(W, shapes):
    if not isinstance(W, dict): raise SandboxError("not_dict", "weights must be a dict")
    out = {}
    for k, shp in shapes.items():
        if k not in W: raise SandboxError("shape", f"missing parameter {k} (got {sorted(W)[:10]})")
        a = np.asarray(W[k], dtype=np.float64)
        if a.shape != tuple(shp) and a.size == int(np.prod(shp)): a = a.reshape(shp)
        if a.shape != tuple(shp): raise SandboxError("shape", f"{k} shape {a.shape} != {tuple(shp)}")
        if not np.isfinite(a).all(): raise SandboxError("nonfinite", f"{k} has non-finite values")
        out[k] = a
    return out

def load_weights(model, W):
    sd = model.state_dict()
    with torch.no_grad():
        for k in sd: sd[k].copy_(torch.tensor(W[k], dtype=sd[k].dtype))
    return model.to(DEVICE)

def get_weights(model): return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}

def margin_stats(lg, y):
    """Raw logit margin (true - best other), margin normalised by per-example logit std (scale-free), and softmax-probability margin."""
    lg = lg.double(); c = lg.gather(1, y[:, None]).squeeze(1); o = lg.clone(); o.scatter_(1, y[:, None], -float("inf"))
    raw = c - o.max(-1).values
    sd = lg.std(-1).clamp_min(1e-12); norm = raw / sd
    p = torch.softmax(lg, -1); pc = p.gather(1, y[:, None]).squeeze(1); po = p.clone(); po.scatter_(1, y[:, None], -1.0)
    return raw, norm, pc - po.max(-1).values

@torch.no_grad()
def evaluate(model, X, Y, idx=None, noise=0.0, seed=0, bs=16384):
    model.eval().to(DEVICE)
    if idx is not None: X, Y = X[idx], Y[idx]
    g = torch.Generator().manual_seed(seed); preds, raws, norms, pms, losses = [], [], [], [], []
    for i in range(0, len(Y), bs):
        x = X[i:i+bs]
        if noise > 0: x = x + noise * torch.randn(x.shape, generator=g)
        x, y = x.to(DEVICE), Y[i:i+bs].to(DEVICE)
        lg = torch.nan_to_num(model(x).float(), nan=0.0, posinf=1e30, neginf=-1e30)
        preds.append(lg.argmax(-1).cpu()); r, n, pm = margin_stats(lg, y)
        raws.append(r.cpu()); norms.append(n.cpu()); pms.append(pm.cpu())
        losses.append(F.cross_entropy(lg.double(), y, reduction="sum").item())
    pred, r, n, pm = torch.cat(preds), torch.cat(raws), torch.cat(norms), torch.cat(pms)
    wrong = (pred != Y.cpu()).nonzero().squeeze(1)
    return {"acc": (pred == Y.cpu()).float().mean().item(), "loss": sum(losses) / len(Y),
            "mean_margin_raw": r.mean().item(), "min_margin_raw": r.min().item(),
            "mean_margin_norm": n.mean().item(), "min_margin_norm": n.min().item(),
            "mean_prob_margin": pm.mean().item(), "min_prob_margin": pm.min().item(),
            "wrong": wrong[:2000].tolist(), "pred": pred}

def fourier_energy(task, W):
    if task not in ("modadd", "modmul", "permadd") or "W1" not in W: return float("nan")
    E = W["W1"][:, :P]
    if task == "modmul": E = E[:, [pow(G_ROOT, e, P) for e in range(P - 1)]]
    if task == "permadd": E = E[:, np.argsort(PERM)]      # columns ordered by underlying value
    pw = (np.abs(np.fft.rfft(E, axis=1)) ** 2).sum(0)[1:]
    return float(np.sort(pw)[::-1][:10].sum() / (pw.sum() + 1e-12))

def structure(task, W):
    out = {}
    for name, A in W.items():
        if A.ndim < 2: continue
        A2 = A.reshape(A.shape[0], -1); s = np.linalg.svd(A2, compute_uv=False)
        if s[0] <= 0: out.update({f"{name}_rank": 0, f"{name}_eff_rank": 0.0, f"{name}_sparsity": 1.0}); continue
        q = s / s.sum()
        out[f"{name}_rank"] = int((s > 1e-3 * s[0]).sum()); out[f"{name}_eff_rank"] = float(np.exp(-(q @ np.log(q + 1e-12))))
        out[f"{name}_sparsity"] = float((np.abs(A2) < 1e-3 * np.abs(A2).max()).mean()); out[f"{name}_fro"] = float(np.linalg.norm(A2))
    out["fourier_top10_energy"] = fourier_energy(task, W)
    return out

NOISES = [0.1, 0.3, 0.5]
MARGIN_KEYS = ["mean_margin_raw", "min_margin_raw", "mean_margin_norm", "min_margin_norm", "mean_prob_margin", "min_prob_margin"]
def full_metrics(task, model, T, sp=None):
    r_all = evaluate(model, T["X"], T["Y"])
    row = {"all_acc": r_all["acc"], **{k: r_all[k] for k in MARGIN_KEYS}}
    if sp is not None: row["test_acc"] = evaluate(model, T["X"], T["Y"], sp[2])["acc"]
    for nz in NOISES: row[f"acc_noise{nz}"] = evaluate(model, T["X"], T["Y"], noise=nz)["acc"]
    row.update(structure(task, get_weights(model))); return row

def t_ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float); n = len(x)
    if n == 0: return np.nan, np.nan, np.nan
    if n == 1: return x[0], 0.0, 0.0
    try:
        from scipy import stats; tq = stats.t.ppf(0.975, n - 1)
    except Exception: tq = 1.96
    return x.mean(), x.std(ddof=1), tq * x.std(ddof=1) / math.sqrt(n)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

PER_RUN, A_CURVES, C_ROWS = [], [], []
def save_all():
    pd.DataFrame(PER_RUN).drop(columns=["pred"], errors="ignore").to_csv(f"{OUT_DIR}/per_run.csv", index=False)
    pd.DataFrame(A_CURVES).to_csv(f"{OUT_DIR}/A_curves.csv", index=False)
    pd.DataFrame(C_ROWS).to_csv(f"{OUT_DIR}/C_edits.csv", index=False)
    json.dump({"per_run": PER_RUN, "C": C_ROWS, "llm_usage": globals().get("USAGE", {}), "elapsed_s": time.time() - T_START},
              open(f"{OUT_DIR}/results_partial.json", "w"), default=str)

ORACLE_W = {}
if RUN_ORACLE_CHECK:
    for t, T in TASKS.items():
        W = validate(run_generated(ORACLE_SRC[t]), T["shapes"]); ORACLE_W[t] = W
        m = load_weights(T["make_model"](), W); r = full_metrics(t, m, T)
        assert r["all_acc"] == 1.0, f"oracle {t} not exact: {r['all_acc']}"
        PER_RUN.append({"task": t, "instance": "B_oracle", "synth": "oracle", "synth_model": "analytic", "prompt": "-", "rep": 0, "ok": True, "success": True, **r})
        print(f"oracle {t}: acc {r['all_acc']:.4f} | norm min margin {r['min_margin_norm']:.3f} | prob margin {r['min_prob_margin']:.4f}")
    save_all()
