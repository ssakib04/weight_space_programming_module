df = pd.DataFrame(PER_RUN).drop(columns=["pred"], errors="ignore")
if "dry_run" in df: df = df[df["dry_run"] != True]     # dry-run mock rows are never aggregated as results
# For A, keep only the LR chosen on validation (per task, optimizer), averaged over seeds.
sel = []
if len(df) and "lr" in df:
    for (t, s), g in df[df.instance == "A"].groupby(["task", "synth"]):
        best_lr = g.groupby("lr")["val_acc"].mean().idxmax(); sel.append(g[g.lr == best_lr])
dfA_sel = pd.concat(sel) if sel else pd.DataFrame()
dfagg_src = pd.concat([df[df.instance != "A"], dfA_sel]) if len(df) else df
agg = []
for keys, g in dfagg_src.groupby(["task", "instance", "synth", "prompt"], dropna=False):
    accs = g["all_acc"].fillna(0).tolist() if keys[1] != "A" else g["test_acc"].tolist()
    mu, sd, ci = t_ci(accs); n = len(g)
    is_cifar = keys[0] == "cifar"
    if is_cifar: k = None; lo = hi = np.nan
    else:
        k = int(g["success"].fillna(False).astype(bool).sum()) if keys[1] == "B" else int(sum(a >= SUCCESS_THRESH for a in accs)); lo, hi = wilson(k, n)
    row = {"task": keys[0], "instance": keys[1], "synth": keys[2], "prompt": keys[3], "n": n, "acc_mean": mu, "acc_std": sd, "acc_ci95": ci,
           "success_rate": (k / n if n else np.nan) if not is_cifar else np.nan,
           "note": (f"accuracy only; beats chance (10%) by >{CIFAR_ABOVE_CHANCE_PTS*100:.0f} pts: {mu - CIFAR_CHANCE > CIFAR_ABOVE_CHANCE_PTS}" if is_cifar else ""),
           "synth_requested": ",".join(sorted(set(map(str, g["synth_requested"].dropna())))) if "synth_requested" in g else "", "success_ci_lo": lo, "success_ci_hi": hi,
           "valid_code_rate": float(g["ok"].astype(bool).mean()) if "ok" in g else np.nan}
    if "error_kind" in g: row["error_kinds"] = json.dumps(g["error_kind"].dropna().value_counts().to_dict())
    if keys[1] == "A": row.update(lr=g["lr"].iloc[0], steps=g["steps"].mean(), compute_s=g["compute_s"].mean())
    for c in ["acc_noise0.1", "acc_noise0.3", "acc_noise0.5", "mean_margin_raw", "min_margin_raw", "mean_margin_norm", "min_margin_norm", "mean_prob_margin", "min_prob_margin", "W1_eff_rank", "W1_sparsity",
              "fourier_top10_energy", "in_tok", "out_tok", "cost_usd", "synth_wall_s"]:
        if c in g: row[c] = pd.to_numeric(g[c], errors="coerce").mean()
    agg.append(row)
dfagg = pd.DataFrame(agg); dfagg.to_csv(f"{OUT_DIR}/comparison_table.csv", index=False)
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 250); dfagg
