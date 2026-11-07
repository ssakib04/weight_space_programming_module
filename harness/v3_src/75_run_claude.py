# ---------------- Claude: capped re-run of selected cases only ----------------
CLAUDE_SYN = {"name": "claude", "kind": "claude", "model_id": CLAUDE_MODEL_ID}
def claude_plan():
    """Per-case plan with cache status and worst-case cost (cached cases cost $0)."""
    rows = []
    for task, level, reps in RUN_CLAUDE_ONLY_CASES:
        if task != "cifar" and task not in TASKS: continue
        user = CIFAR_PROMPT if task == "cifar" else task_prompt(task, level)
        mt = CLAUDE_MAX_TOKENS["cifar" if task == "cifar" else "algo"]
        for rep in reps:
            hit = cache_get(CLAUDE_MODEL_ID, SYSTEM_PROMPT, user, rep, mt) is not None
            rows.append({"task": task, "prompt": level, "rep": rep, "cached": hit, "section": "B_cifar" if task == "cifar" else "B_algo",
                         "worst_usd": 0.0 if hit else worst_case_cost(user, SYSTEM_PROMPT, mt)})
    return pd.DataFrame(rows)

PLAN = claude_plan()
if len(PLAN):
    print(PLAN.groupby(["task", "prompt", "section"]).agg(n=("rep", "size"), cached=("cached", "sum"), worst_usd=("worst_usd", "sum")))
    print("PRE-RUN ESTIMATE: worst-case total $%.2f | caps: total $%.2f, sections %s | DRY_RUN_CLAUDE=%s RUN_CLAUDE=%s"
          % (PLAN.worst_usd.sum(), CLAUDE_BUDGET_USD, CLAUDE_SECTION_CAP, DRY_RUN_CLAUDE, RUN_CLAUDE))
if RUN_CLAUDE:
    st = []
    try:
        for _, p in PLAN.iterrows():
            if p.task == "cifar":
                if CIFAR_DATA is not None: cifar_once(CLAUDE_SYN, int(p.rep))
            else:
                row = synth_once(CLAUDE_SYN, p.task, p.prompt, int(p.rep))
                PER_RUN.append(row); save_all()
                print(f"[B claude {p.task} {p.prompt} r{p.rep}] src={row.get('reply_source')} ok={row['ok']} acc={row['all_acc']:.4f} {row.get('error_kind') or ''}")
        st.append("done")
    except BudgetExceeded as e: st.append("BUDGET_STOP"); print("BUDGET STOP:", e)
    SYN_STATUS["claude"] = {"status": st, "spent_usd": BUDGET.spent, "by_section": BUDGET.by_section, "api_calls": API_CALLS["claude"]}
    save_all()
print("Claude: real API calls %d | spent $%.4f" % (API_CALLS["claude"], BUDGET.spent))
