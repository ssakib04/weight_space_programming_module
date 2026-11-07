def error_report(task, model, T):
    r = evaluate(model, T["X"], T["Y"]); W = get_weights(model); wrong = r["wrong"]
    X, Y = T["X"], T["Y"]
    rep = {"acc": r["acc"], "n_wrong": int((r["pred"] != Y).sum()), "min_margin_norm": r["min_margin_norm"], "min_prob_margin": r["min_prob_margin"],
           "dead_hidden_units": int((np.abs(W["W1"]).sum(1) < 1e-8).sum()),
           "param_norms": {k: float(np.linalg.norm(v)) for k, v in W.items()}}
    if task in ("modadd", "modmul", "permadd"):
        ab = [(int(X[i, :P].argmax()), int(X[i, P:].argmax()), int(Y[i]), int(r["pred"][i])) for i in wrong[:20]]
        rep["sample_wrong (a,b,true,pred)"] = ab; rep["fourier_top10_energy"] = fourier_energy(task, W)
    else:
        rep["sample_wrong (bits,true,pred)"] = [("".join(str(int(v)) for v in X[i].tolist()), int(Y[i]), int(r["pred"][i])) for i in wrong[:20]]
    return rep

def damage_weights(W, seed=0):
    g = np.random.default_rng(seed); W = {k: v.copy() for k, v in W.items()}; H = W["W1"].shape[0]
    dead = g.choice(H, int(0.8 * H), replace=False); W["W1"][dead] = 0; W["W2"][:, dead] = 0
    if "b1" in W: W["b1"][dead] = 0
    W["W1"] += 0.5 * g.standard_normal(W["W1"].shape); return W

C_PROMPT = """You are repairing the weights of the network below WITHOUT gradients or training.
{spec}
The current weights are available to your code as a dict W of numpy arrays (same names/shapes as above).
Error report of the current weights (JSON): {report}
Return ONLY one python code block defining `def edit_weights(W):` that returns a dict with ALL parameters (same names and shapes).
Use only numpy (np) and math. You may rewrite any part of the weights."""

def run_C():
    claude = CLAUDE_SYN
    deadline = time.time() + TIME_LIMIT["C"]
    for task, T in TASKS.items():
        fails = [r for r in PER_RUN if r.get("instance") == "B" and r["task"] == task and r["synth"] == label_of(claude)
                 and r["ok"] and not r["success"]][:C_MAX_TARGETS_PER_TASK]
        targets = [("damaged_oracle", damage_weights(ORACLE_W[task]))]
        for r in fails:
            f = f"{GEN_DIR}/{label_of(claude)}_{task}_{r['prompt']}_r{r['rep']}_weights.npz"
            if os.path.exists(f): targets.append((f"B_{r['prompt']}_r{r['rep']}", dict(np.load(f))))
        for name, W0 in targets:
            if time.time() > deadline: print("C time limit"); return
            cur = W0; m = load_weights(T["make_model"](), cur); acc0 = evaluate(m, T["X"], T["Y"])["acc"]; best = (acc0, cur)
            row = {"task": task, "target": name, "acc_before": acc0, "iters": [], "dry_run": DRY_RUN_CLAUDE}
            for it in range(C_ITERS):
                try:
                    rep = error_report(task, m, T)
                    text, meta = call_llm(claude, SYSTEM_PROMPT, C_PROMPT.format(spec=task_prompt(task, "full"), report=json.dumps(rep, default=str)),
                                          f"C_{task}_{name}_it{it}", rep=it, section="C", max_tokens=CLAUDE_MAX_TOKENS["C"])
                    code_str, _ = extract_code(text, "edit_weights", meta.get("stop_reason")); open(f"{GEN_DIR}/C_{task}_{name}_it{it}.py", "w").write(code_str)
                    cur = validate(run_generated(code_str, "edit_weights", inputs=cur), T["shapes"])
                    m = load_weights(T["make_model"](), cur); a = evaluate(m, T["X"], T["Y"])["acc"]
                    row["iters"].append({"it": it, "acc": a, **meta})
                    if a > best[0]: best = (a, cur)
                    if a >= 0.9999: break
                except BudgetExceeded as e: row["iters"].append({"it": it, "error": "BUDGET_STOP"}); break
                except Exception as e: row["iters"].append({"it": it, "error": repr(e)[:300]})
            row["acc_after"] = best[0]; C_ROWS.append(row); save_all()
            print(f"[C {task} {name}] {acc0:.4f} -> {best[0]:.4f}")
if RUN_C_CLAUDE and RUN_CLAUDE:
    try: run_C()
    except Exception as e: print("C section failed:", repr(e)); traceback.print_exc()
