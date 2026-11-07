def train_A_algo(task, T, opt_name, lr, seed, t_deadline):
    set_seed(seed); sp = splits(len(T["Y"]), seed); m = T["make_model"]().to(DEVICE)
    if opt_name == "adamw": opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY["adamw"], betas=(0.9, 0.98))
    else: opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY["sgd"])
    xtr, ytr = T["X"][sp[0]].to(DEVICE), T["Y"][sp[0]].to(DEVICE); gpu_s = 0.0; rows = []
    for step in range(1, max(CHECKPOINTS) + 1):
        t0 = time.perf_counter(); m.train()
        loss = F.cross_entropy(m(xtr), ytr); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if DEVICE == "cuda": torch.cuda.synchronize()
        gpu_s += time.perf_counter() - t0
        if step in CHECKPOINTS:
            rows.append({"task": task, "opt": opt_name, "lr": lr, "seed": seed, "step": step, "train_loss": loss.item(),
                         "val_acc": evaluate(m, T["X"], T["Y"], sp[1])["acc"], "test_acc": evaluate(m, T["X"], T["Y"], sp[2])["acc"],
                         "train_compute_s": gpu_s})
        if time.time() > t_deadline: break
    return m, rows, sp

def run_A_algo():
    deadline = time.time() + TIME_LIMIT["A_algo"]
    for t, T in TASKS.items():
        for opt_name, lrs in LR_GRID.items():
            best = None
            for lr in lrs:
                for seed in SEEDS_A:
                    if time.time() > deadline: print("A_algo time limit hit"); return
                    try:
                        m, rows, sp = train_A_algo(t, T, opt_name, lr, seed, deadline)
                        A_CURVES.extend(rows); last = rows[-1]
                        r = full_metrics(t, m, T, sp)
                        PER_RUN.append({"task": t, "instance": "A", "synth": f"{opt_name}", "prompt": "-", "rep": seed, "lr": lr,
                                        "steps": last["step"], "val_acc": last["val_acc"], "compute_s": last["train_compute_s"], "ok": True, **r})
                        print(f"[A {t} {opt_name} lr={lr} s={seed}] val {last['val_acc']:.3f} test {r['test_acc']:.3f} ({last['train_compute_s']:.0f}s)")
                    except Exception as e:
                        print("A run failed:", t, opt_name, lr, seed, repr(e)); traceback.print_exc()
                    save_all()
if RUN_A and TASKS:
    try: run_A_algo()
    except Exception as e: print("A_algo section failed:", repr(e))
