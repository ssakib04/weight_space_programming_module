# ---- Final results.json + zip ----
results = {"config": {k: globals()[k] for k in ["QUICK", "MOCK", "RUN_TASKS", "PROMPT_LEVELS", "N_REPEATS_LOCAL",
           "N_REPEATS_CIFAR_LOCAL", "SUCCESS_THRESH", "CLAUDE_TEMPERATURE", "CLAUDE_MAX_TOKENS", "PRICE_PER_MTOK", "CLAUDE_BUDGET_USD",
           "CLAUDE_SECTION_CAP", "RUN_CLAUDE", "DRY_RUN_CLAUDE", "RUN_CLAUDE_ONLY_CASES", "CLAUDE_MODEL_ID",
           "LOCAL_MAX_NEW_TOKENS", "LOCAL_DETERMINISTIC", "LOCAL_TEMPERATURE", "LOCAL_TOP_P", "LOCAL_SEED_BASE", "SEEDS_A", "LR_GRID", "WEIGHT_DECAY", "CHECKPOINTS", "SPLIT",
           "CIFAR_STEPS", "CIFAR_LR_GRID", "P", "H_ALGO", "PAR_N", "PAR_K", "PAR_SUBSET", "G_ROOT"]}, "PERM": PERM.tolist(),
           "synthesizers": SYNTHESIZERS, "synth_status": SYN_STATUS, "llm_usage": USAGE, "claude_spend_usd": BUDGET.spent, "claude_spend_by_section": BUDGET.by_section, "claude_api_calls": API_CALLS["claude"],
           "env": {"torch": torch.__version__, "python": platform.python_version(), "device": DEVICE,
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
           "elapsed_s": time.time() - T_START, "comparison": json.loads(dfagg.to_json(orient="records")),
           "per_run": json.loads(pd.DataFrame(PER_RUN).drop(columns=["pred"], errors="ignore").to_json(orient="records")), "C": C_ROWS}
json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=1, default=str)
import shutil
_z = shutil.make_archive(os.path.join(tempfile.gettempdir(), "wsp_v3_outputs"), "zip", OUT_DIR)
zip_path = shutil.move(_z, os.path.join(OUT_DIR, "wsp_v3_outputs.zip"))
print("elapsed %.1f min | Claude $%.3f | zip: %s" % ((time.time() - T_START) / 60, BUDGET.spent, zip_path))
