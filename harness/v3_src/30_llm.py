# ---------------- LLM layer: budget guard, reply cache, Claude (dry-run by default), local HF models, mock ----------------
LLM_LOG = f"{OUT_DIR}/llm_calls.jsonl"
USAGE = {}
API_CALLS = {"claude": 0}          # number of REAL Anthropic API requests made in this process
class BudgetExceeded(Exception): pass

class Budget:
    def __init__(self, total, section_caps):
        self.total, self.caps, self.spent, self.by_section = float(total), dict(section_caps), 0.0, {}
    def check(self, section, worst):
        s = self.by_section.get(section, 0.0); cap = self.caps.get(section, self.total)
        if self.spent + worst > self.total: raise BudgetExceeded(f"total cap: spent ${self.spent:.3f} + worst ${worst:.3f} > ${self.total:.2f}")
        if s + worst > cap: raise BudgetExceeded(f"section {section} cap: spent ${s:.3f} + worst ${worst:.3f} > ${cap:.2f}")
    def add(self, section, cost):
        self.spent += cost; self.by_section[section] = self.by_section.get(section, 0.0) + cost
BUDGET = Budget(CLAUDE_BUDGET_USD, CLAUDE_SECTION_CAP)

def worst_case_cost(user, system, max_tokens):
    return ((len(user) + len(system)) / 3.0 * PRICE_PER_MTOK["input"] + max_tokens * PRICE_PER_MTOK["output"]) / 1e6

def _secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)

def _log(rec):
    with open(LLM_LOG, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
def _u(name):
    return USAGE.setdefault(name, {"calls": 0, "cached": 0, "input": 0, "output": 0, "cost_usd": 0.0, "latency_s": 0.0, "errors": 0})

# ---- reply cache keyed by (model, sha256(system+user), repeat idx) ----
def cache_key(model, system, user, rep):
    h = hashlib.sha256(json.dumps([system, user]).encode()).hexdigest()[:20]
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}__{h}__r{rep}"
def cache_get(model, system, user, rep, max_tokens):
    f = os.path.join(CACHE_DIR, cache_key(model, system, user, rep) + ".json")
    if not os.path.exists(f): return None
    c = json.load(open(f))
    if c.get("stop_reason") == "max_tokens" and c.get("max_tokens", 0) < max_tokens: return None   # truncated at a lower cap -> re-ask
    return c
def cache_put(model, system, user, rep, rec):
    f = os.path.join(CACHE_DIR, cache_key(model, system, user, rep) + ".json")
    json.dump(rec, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)

def seed_cache_from_v2(path, max_tokens_v2=6000):
    """Import Claude replies saved by v2 (llm_calls.jsonl) into the cache. Tag format: B_<task>_<level>_r<rep> or B_cifar_r<rep>."""
    if not path or not os.path.exists(path): return 0
    n = 0
    for line in open(path):
        r = json.loads(line)
        if not str(r.get("synth", "")).startswith("claude") or "response" not in r or "user" not in r: continue
        m = re.match(r"B_(\w+?)_(?:(full|partial|none)_)?r(\d+)$", r.get("tag", ""))
        if not m: continue
        rep = int(m.group(3)); model = r.get("model_requested") or CLAUDE_MODEL_ID
        if cache_get(model, r["system"], r["user"], rep, 0) is None:
            cache_put(model, r["system"], r["user"], rep, {"response": r["response"], "stop_reason": r.get("stop_reason"),
                      "max_tokens": max_tokens_v2, "usage": r.get("usage"), "model_returned": r.get("model_returned"), "source": "v2_log", "tag": r["tag"]})
            n += 1
    return n

def _default_client_factory():
    key = _secret("ANTHROPIC_API_KEY")
    if not key: raise RuntimeError("No ANTHROPIC_API_KEY secret")
    import anthropic
    return anthropic.Anthropic(api_key=key)
CLIENT_FACTORY = _default_client_factory
_CLAUDE = {}

def call_claude(syn, system, user, tag, rep=0, section="B_algo", max_tokens=6000):
    model = syn["model_id"]; u = _u(syn["name"])
    c = cache_get(model, system, user, rep, max_tokens)
    if c is not None:
        u["cached"] += 1
        _log({"tag": tag, "synth": syn["name"], "event": "CACHE_HIT", "source": c.get("source"), "key": cache_key(model, system, user, rep)})
        return c["response"], {"in_tok": 0, "out_tok": 0, "cost_usd": 0.0, "latency_s": 0.0, "cached": True,
                               "stop_reason": c.get("stop_reason"), "reply_source": c.get("source", "cache")}
    if DRY_RUN_CLAUDE:            # never builds a client, never spends
        text, meta = call_mock(syn, system, user, tag)
        _log({"tag": tag, "synth": syn["name"], "event": "DRY_RUN_MOCK"})
        return text, {**meta, "cached": False, "stop_reason": "dry_run", "reply_source": "dry_run_mock"}
    worst = worst_case_cost(user, system, max_tokens)
    try: BUDGET.check(section, worst)
    except BudgetExceeded as e:
        _log({"tag": tag, "synth": syn["name"], "event": "BUDGET_STOP", "spent": BUDGET.spent, "why": str(e)}); raise
    if "client" not in _CLAUDE: _CLAUDE["client"] = CLIENT_FACTORY()
    client = _CLAUDE["client"]
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
    t0 = time.perf_counter(); temp = CLAUDE_TEMPERATURE
    API_CALLS["claude"] += 1
    if _CLAUDE.get("no_temp"): temp = None; resp = client.messages.create(**kw)
    else:
        try: resp = client.messages.create(temperature=CLAUDE_TEMPERATURE, **kw)
        except Exception as e:
            if "temperature" not in str(e).lower(): raise
            _CLAUDE["no_temp"] = True; temp = None; API_CALLS["claude"] += 1; resp = client.messages.create(**kw)
    dt = time.perf_counter() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    ui, uo = resp.usage.input_tokens, resp.usage.output_tokens
    cost = (ui * PRICE_PER_MTOK["input"] + uo * PRICE_PER_MTOK["output"]) / 1e6; BUDGET.add(section, cost)
    u["calls"] += 1; u["input"] += ui; u["output"] += uo; u["cost_usd"] += cost; u["latency_s"] += dt
    rec = {"response": text, "stop_reason": resp.stop_reason, "max_tokens": max_tokens, "usage": {"input": ui, "output": uo},
           "model_returned": resp.model, "source": "api", "tag": tag, "temperature": temp}
    cache_put(model, system, user, rep, rec)
    _log({"tag": tag, "synth": syn["name"], "model_requested": model, "model_returned": resp.model, "temperature": temp, "system": system,
          "user": user, "response": text, "usage": {"input": ui, "output": uo}, "cost_usd": cost, "cum_cost_usd": BUDGET.spent,
          "section": section, "latency_s": dt, "stop_reason": resp.stop_reason, "max_tokens": max_tokens, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if resp.stop_reason == "max_tokens": print(f"  WARNING {tag}: reply truncated at max_tokens={max_tokens}")
    return text, {"in_tok": ui, "out_tok": uo, "cost_usd": cost, "latency_s": dt, "cached": False, "stop_reason": resp.stop_reason, "reply_source": "api"}

# ---- local HF models ----
_HF = {}
def load_hf(syn):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokn = _secret("HF_TOKEN"); tok_kw = {"token": tokn} if tokn else {}
    if syn.get("requires_hf_token") and not tokn:
        raise RuntimeError(f"SKIPPED {syn['model_id']}: gated model and no HF_TOKEN secret (no silent fallback in v3)")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(syn["model_id"], **tok_kw)
    mdl = AutoModelForCausalLM.from_pretrained(syn["model_id"], quantization_config=q, device_map="auto", torch_dtype=torch.float16, **tok_kw)
    mdl.eval(); _HF.update(tok=tok, model=mdl, id=syn["model_id"]); syn["resolved_id"] = syn["model_id"]

def unload_hf():
    _HF.clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

@torch.no_grad()
def call_hf(syn, system, user, tag, rep=0, **_):
    tok, mdl = _HF["tok"], _HF["model"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to(mdl.device); seed = LOCAL_SEED_BASE + rep; set_seed(seed)
    gen_kw = dict(max_new_tokens=LOCAL_MAX_NEW_TOKENS, pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(ids))
    gen_kw.update(do_sample=False) if LOCAL_DETERMINISTIC else gen_kw.update(do_sample=True, temperature=LOCAL_TEMPERATURE, top_p=LOCAL_TOP_P)
    t0 = time.perf_counter(); out = mdl.generate(ids, **gen_kw)
    dt = time.perf_counter() - t0; gen = out[0, ids.shape[1]:]; text = tok.decode(gen, skip_special_tokens=True)
    stop = "max_tokens" if len(gen) >= LOCAL_MAX_NEW_TOKENS else "end_turn"
    u = _u(syn["name"]); u["calls"] += 1; u["input"] += ids.shape[1]; u["output"] += len(gen); u["latency_s"] += dt
    _log({"tag": tag, "synth": syn["name"], "model_requested": syn["model_id"], "model_returned": _HF["id"], "seed": seed,
          "do_sample": not LOCAL_DETERMINISTIC, "temperature": None if LOCAL_DETERMINISTIC else LOCAL_TEMPERATURE, "system": system, "user": user,
          "response": text, "usage": {"input": int(ids.shape[1]), "output": int(len(gen))}, "stop_reason": stop, "latency_s": dt})
    return text, {"in_tok": int(ids.shape[1]), "out_tok": int(len(gen)), "cost_usd": 0.0, "latency_s": dt, "stop_reason": stop, "reply_source": "local"}

# ---- mock (offline) ----
_MOCK_RNG = np.random.default_rng(0)
def call_mock(syn, system, user, tag, **_):
    task = next((t for t in ORACLE_SRC if f"[[task={t}]]" in user), None)
    if task is None: task = next((t for t, k in [("permadd", "SECRET relabelling"), ("modmul", "(a*b) mod p"), ("modadd", "(a+b) mod p"),
                                                 ("parity", "parity (XOR)")] if k in user), None)
    r = _MOCK_RNG.random()
    if tag.startswith("C_"):
        text = "Here is the edit.\n```python\nimport numpy as np\ndef edit_weights(W):\n    W = dict(W)\n    W['W2'] = W['W2'] * 1.0\n    return W\n```"
    elif "CIFAR-10" in user:
        text = "```python\nimport numpy as np\ndef build_weights():\n    r = np.random.default_rng(0)\n    return {k: r.standard_normal(s)*0.1 for k, s in " + repr(CIFAR_SHAPES) + ".items()}\n```"
    elif task and r < 0.7:
        text = "Sure, here is the construction.\n```python\nimport numpy as np\nimport math\n" + ORACLE_SRC[task] + "```\nThis works because..."
    elif r < 0.85:
        text = "```python\ndef build_weights():\n    return {'W1': np.zeros((3,3))}\n```"
    else:
        text = "no code here"
    _u(syn["name"])["calls"] += 1
    _log({"tag": tag, "synth": syn["name"], "event": "MOCK", "response": text[:300]})
    return text, {"in_tok": 0, "out_tok": 0, "cost_usd": 0.0, "latency_s": 0.0, "stop_reason": "mock", "reply_source": "mock"}

def call_llm(syn, system, user, tag, rep=0, section="B_algo", max_tokens=6000):
    k = syn["kind"]
    try:
        if k == "claude": return call_claude(syn, system, user, tag, rep=rep, section=section, max_tokens=max_tokens)
        return {"hf": call_hf, "mock": call_mock}[k](syn, system, user, tag, rep=rep)
    except BudgetExceeded: raise
    except Exception:
        _u(syn["name"])["errors"] += 1; raise

N_SEEDED = seed_cache_from_v2(V2_LLM_LOG)
print("cache dir", CACHE_DIR, "| seeded from v2 log:", N_SEEDED, "| entries:", len(os.listdir(CACHE_DIR)))
