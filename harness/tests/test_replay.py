"""(a) Replay every saved v2 reply through the v3 extraction+sandbox+evaluation and compare with v2 outcomes."""
import sys, os, json, re, collections; sys.path.insert(0, os.path.dirname(__file__)); from common import load
import pandas as pd
real = os.environ.get("REAL_CIFAR") == "1"
ns = load(env={"WSP_FAKE_CIFAR": "0" if real else "1", "WSP_QUICK": "0" if real else "1"})
V2 = os.path.join(os.path.dirname(__file__), "..", "v2_outputs")
old = pd.read_csv(f"{V2}/per_run.csv"); old = old[old.instance == "B"]
recs = [json.loads(l) for l in open(f"{V2}/llm_calls.jsonl")]
rows = []
for r in recs:
    m = re.match(r"B_(\w+?)_(?:(full|partial|none)_)?r(\d+)$", r.get("tag", ""))
    if not m: continue
    task, level, rep = m.group(1), m.group(2) or "full", int(m.group(3))
    o = old[(old.synth == r["synth"]) & (old.task == task) & (old.prompt == level) & (old.rep == rep)]
    o = o.iloc[0] if len(o) else None
    new = ns["process_reply"](task, r["response"], {"stop_reason": r.get("stop_reason")}, f"replay_{r['synth']}_{task}_{level}_r{rep}",
                              ns["CIFAR_SHAPES"] if task == "cifar" else None, cifar=(task == "cifar"))
    new.pop("_model", None)
    old_ok = bool(o is not None and o.ok); old_acc = float(o.all_acc) if o is not None and pd.notna(o.all_acc) else 0.0
    old_err = "" if o is None or pd.isna(o.error) else str(o.error)
    old_kind = "" if not old_err else ("name_error_next" if "name 'next'" in old_err else "fence_syntax" if "```" in old_err else
               "disallowed_token" if "disallowed token" in old_err else ns["classify_error"](old_err))
    if o is None and task == "cifar": old_kind = "no_row(v2 CIFAR B rows absent: exception before append)"
    rows.append({"synth": r["synth"], "task": task, "prompt": level, "rep": rep, "stop_reason": r.get("stop_reason"),
                 "old_ok": old_ok, "old_acc": old_acc, "old_success": old_ok and old_acc >= 0.99 and task != "cifar", "old_err_kind": old_kind,
                 "new_ok": new["ok"], "new_acc": new["all_acc"], "new_success": new.get("success"), "new_err_kind": new.get("error_kind"),
                 "extract_method": new.get("extract_method"), "new_err": (new.get("error") or "")[:160]})
df = pd.DataFrame(rows); out = os.path.join(os.path.dirname(__file__), "..", "v3_test_outputs"); os.makedirs(out, exist_ok=True)
df.to_csv(f"{out}/replay_{'realcifar' if real else 'fakecifar'}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 120)
algo = df[df.task != "cifar"]
print(algo.groupby(["synth", "task"]).agg(n=("rep", "size"), old_success=("old_success", "sum"), new_success=("new_success", "sum"),
                                          old_valid=("old_ok", "sum"), new_valid=("new_ok", "sum")))
print(algo[algo.synth.str.startswith("claude")].groupby(["task", "prompt"]).agg(n=("rep", "size"), old=("old_success", "sum"), new=("new_success", "sum")))
print("\nclaude old failure kinds -> new outcome:")
c = algo[algo.synth.str.startswith("claude") & ~algo.old_success]
print(c.groupby(["old_err_kind", "new_success"]).size())
print(c[~c.new_success.astype(bool)][["task", "prompt", "rep", "stop_reason", "old_err_kind", "new_err_kind", "new_acc", "new_err"]].to_string())
print("\nregressions (old success, new not):", int((algo.old_success & ~algo.new_success.astype(bool)).sum()))
print("\nCIFAR:"); print(df[df.task == "cifar"][["synth", "rep", "stop_reason", "old_err_kind", "new_ok", "new_acc", "new_err_kind", "extract_method", "new_err"]].to_string())
