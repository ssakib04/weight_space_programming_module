# ---------------- Instance B: one synthesis attempt -> extraction -> sandbox -> validation -> metrics ----------------
def label_of(syn):
    """Row label = model that actually produced the reply (never the requested-but-unavailable one)."""
    return syn.get("resolved_id") or syn["model_id"]

def process_reply(task, text, meta, fn_base, shapes=None, cifar=False):
    """Shared by live runs and offline replay. Returns row fields (ok, all_acc, error, error_kind, extraction info, metrics)."""
    row = {"ok": False, "all_acc": 0.0, "error": None, "error_kind": None}
    code_str, info = extract_code(text, "build_weights", meta.get("stop_reason"))
    row.update({f"extract_{k}": v for k, v in info.items()})
    open(f"{GEN_DIR}/{fn_base}.py", "w").write(code_str)
    try:
        if not re.search(r"def\s+build_weights\s*\(", code_str):
            raise SandboxError("truncated" if info["truncated"] else "no_code", "no build_weights definition in reply")
        W = validate(run_generated(code_str, timeout=SANDBOX_TIMEOUT["cifar" if cifar else "algo"]), shapes or TASKS[task]["shapes"])
        if cifar:
            m = load_weights(SmallCNN(), W); row.update(cifar_metrics(m, CIFAR_DATA[2], CIFAR_DATA[3])); row["ok"] = True
            row["_model"] = m
        else:
            T = TASKS[task]; m = load_weights(T["make_model"](), W); row.update(full_metrics(task, m, T)); row["ok"] = True
        np.savez_compressed(f"{GEN_DIR}/{fn_base}_weights.npz", **W)
    except SandboxError as e:
        row["error"], row["error_kind"] = str(e)[:500], e.kind
    except Exception as e:
        row["error"], row["error_kind"] = repr(e)[:500], classify_error(repr(e))
    if info["truncated"] and not row["ok"]: row["error_kind"] = "truncated"
    if cifar: row["success"] = None           # accuracy only for CIFAR
    else: row["success"] = bool(row["ok"] and row["all_acc"] >= SUCCESS_THRESH)
    return row

def synth_once(syn, task, level, rep, section="B_algo"):
    user = task_prompt(task, level) + (f"\n[[task={task}]]" if syn["kind"] == "mock" else "")
    tag = f"B_{task}_{level}_r{rep}"; t0 = time.perf_counter()
    base = {"task": task, "instance": "B", "synth": label_of(syn), "synth_requested": syn["model_id"], "synth_kind": syn["kind"],
            "prompt": level, "rep": rep}
    try:
        text, meta = call_llm(syn, SYSTEM_PROMPT, user, tag, rep=rep, section=section, max_tokens=CLAUDE_MAX_TOKENS["algo"])
    except BudgetExceeded: raise
    except Exception as e:
        return {**base, "ok": False, "all_acc": 0.0, "success": False, "error": repr(e)[:400], "error_kind": "llm_call", "synth_wall_s": time.perf_counter() - t0}
    row = {**base, **{k: v for k, v in meta.items()}, **process_reply(task, text, meta, f"{re.sub('[^A-Za-z0-9_.-]', '_', label_of(syn))}_{task}_{level}_r{rep}")}
    row["dry_run"] = meta.get("reply_source") in ("dry_run_mock",)   # dry-run rows are excluded from all tables
    row["synth_wall_s"] = time.perf_counter() - t0
    return row

def run_B_local(syn):
    reps = 1 if LOCAL_DETERMINISTIC else N_REPEATS_LOCAL
    deadline = time.time() + TIME_LIMIT["B_hf_each"]
    for rep in range(reps):
        for task in TASKS:
            for level in PROMPT_LEVELS:
                if time.time() > deadline: print(f"[{label_of(syn)}] time limit hit"); return "timeout"
                row = synth_once(syn, task, level, rep); PER_RUN.append(row); save_all()
                print(f"[B {label_of(syn)} {task} {level} r{rep}] ok={row['ok']} acc={row['all_acc']:.4f} {row.get('error_kind') or ''}")
    return "done"
