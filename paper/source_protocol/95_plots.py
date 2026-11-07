# ---- Plots ----
try:
    dfc = pd.DataFrame(A_CURVES)
    tasks_all = sorted(set(dfagg.task)) if len(dfagg) else []
    if len(dfc):
        fig, axs = plt.subplots(1, len(tasks_all), figsize=(5 * len(tasks_all), 4), squeeze=False)
        for ax, t in zip(axs[0], tasks_all):
            g = dfc[dfc.task == t]
            for (o, lr), gg in g.groupby(["opt", "lr"]):
                mm = gg.groupby("step")[["val_acc", "train_compute_s"]].mean(); ax.plot(mm.index, mm.val_acc, "o-", ms=3, label=f"A {o} lr={lr}")
            for _, r in dfagg[(dfagg.task == t) & (dfagg.instance.isin(["B", "B+ridge"]))].iterrows():
                if r.prompt in ("full", "-") and np.isfinite(r.acc_mean): ax.axhline(r.acc_mean, ls=":", lw=1, label=f"{r.instance} {r.synth} {r.prompt}")
            ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.legend(fontsize=5)
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/accuracy_vs_steps.png", dpi=150); plt.show()
    b = dfagg[(dfagg.instance == "B") & (dfagg.task != "cifar")]
    if len(b):
        fig, axs = plt.subplots(1, len(set(b.task)), figsize=(5 * len(set(b.task)), 4), squeeze=False)
        for ax, (t, g) in zip(axs[0], b.groupby("task")):
            piv = g.pivot_table(index="prompt", columns="synth", values="success_rate")
            piv.plot.bar(ax=ax); ax.set_title(f"B success rate: {t}"); ax.set_ylim(0, 1.05)
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/B_success_by_prompt.png", dpi=150); plt.show()
    fig, axs = plt.subplots(1, max(1, len(tasks_all)), figsize=(5 * max(1, len(tasks_all)), 4), squeeze=False)
    for ax, t in zip(axs[0], tasks_all):
        for _, r in dfagg[dfagg.task == t].iterrows():
            if r.instance == "B" and r.prompt not in ("full",): continue
            ys = [r.acc_mean] + [r.get(f"acc_noise{n}", np.nan) for n in NOISES]; ax.plot([0] + NOISES, ys, "o-", ms=3, label=f"{r.instance} {r.synth}")
        ax.set_title(f"robustness: {t}"); ax.set_xlabel("input noise std"); ax.legend(fontsize=5)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/robustness.png", dpi=150); plt.show()
except Exception as e: print("plotting failed:", repr(e)); traceback.print_exc()
