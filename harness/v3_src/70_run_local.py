# ---------------- Local open-weight synthesizers (no API cost). One model loaded at a time. ----------------
SYN_STATUS = {}
if RUN_LOCAL:
    for syn in SYNTHESIZERS:
        status = []
        try:
            if syn["kind"] == "hf": load_hf(syn)
            if TASKS: status.append("algo:" + run_B_local(syn))
            if CIFAR_DATA is not None:
                for rep in range(1 if LOCAL_DETERMINISTIC else N_REPEATS_CIFAR_LOCAL): cifar_once(syn, rep)
                status.append("cifar:done")
        except Exception as e: status.append("FAILED/SKIPPED " + repr(e)[:200]); print(f"synthesizer {syn['model_id']}:", repr(e)[:300])
        finally:
            if syn["kind"] == "hf": unload_hf()
        SYN_STATUS[syn["model_id"]] = {"label_in_tables": label_of(syn) if syn.get("resolved_id") or syn["kind"] != "hf" else None,
                                       "status": status}
        save_all()
print(json.dumps(SYN_STATUS, indent=1))
