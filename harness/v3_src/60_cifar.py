CIFAR_SHAPES = {"conv1.weight": (32, 3, 5, 5), "conv1.bias": (32,), "conv2.weight": (64, 32, 3, 3), "conv2.bias": (64,),
                "fc.weight": (10, 1024), "fc.bias": (10,)}
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2); self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.fc = nn.Linear(1024, 10)
    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)          # 32x16x16
        x = F.adaptive_avg_pool2d(F.relu(self.conv2(x)), 4)  # 64x4x4
        return x.flatten(1)
    def forward(self, x): return self.fc(self.features(x))

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

def _cifar_parquet(root, split):
    """Fallback: the same CIFAR-10 images from the Hugging Face mirror (uoft-cs/cifar10), decoded with PIL."""
    import io, urllib.request
    from PIL import Image
    os.makedirs(root, exist_ok=True); f = os.path.join(root, f"{split}.parquet")
    if not os.path.exists(f):
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet", f)
    d = pd.read_parquet(f)
    x = np.stack([np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")) for v in d["img"]])
    return torch.tensor(x).permute(0, 3, 1, 2).float() / 255., torch.tensor(d["label"].values.astype(np.int64))

def load_cifar():
    import torchvision, torchvision.transforms as tf
    root = "/kaggle/temp/cifar" if ON_KAGGLE else "./cifar_data"
    try:
        if os.environ.get("WSP_FAKE_CIFAR") == "1": raise RuntimeError("WSP_FAKE_CIFAR=1")
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True); te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        to = lambda d: (torch.tensor(d.data).permute(0, 3, 1, 2).float() / 255., torch.tensor(d.targets))
        (xtr, ytr), (xte, yte) = to(tr), to(te); src = "CIFAR-10"
        if QUICK: xtr, ytr, xte, yte = xtr[:3000], ytr[:3000], xte[:1000], yte[:1000]; src = "CIFAR-10-subset"
    except Exception as e0:
        print("torchvision CIFAR-10 failed:", repr(e0)[:160])
        try:
            if os.environ.get("WSP_FAKE_CIFAR") == "1": raise e0
            xtr, ytr = _cifar_parquet(root, "train"); xte, yte = _cifar_parquet(root, "test"); src = "CIFAR-10(hf-parquet)"
            if QUICK: xtr, ytr, xte, yte = xtr[:3000], ytr[:3000], xte[:1000], yte[:1000]; src += "-subset"
            m = torch.tensor(CIFAR_MEAN)[None, :, None, None]; s = torch.tensor(CIFAR_STD)[None, :, None, None]
            return (xtr - m) / s, ytr, (xte - m) / s, yte, src
        except Exception as e:
            print("CIFAR-10 unavailable -> synthetic stand-in (pipeline test only):", repr(e)[:120])
            g = torch.Generator().manual_seed(0); xtr = torch.rand(2000, 3, 32, 32, generator=g); ytr = torch.randint(0, 10, (2000,), generator=g)
            xte = torch.rand(500, 3, 32, 32, generator=g); yte = torch.randint(0, 10, (500,), generator=g); src = "FAKE"
    m = torch.tensor(CIFAR_MEAN)[None, :, None, None]; s = torch.tensor(CIFAR_STD)[None, :, None, None]
    return (xtr - m) / s, ytr, (xte - m) / s, yte, src

CIFAR_PROMPT = f"""Write the weights of a small CNN for CIFAR-10 classification WITHOUT any training, data, or gradients.
Input: 3x32x32 RGB images normalized per channel with mean {CIFAR_MEAN} and std {CIFAR_STD}. Classes (index order): {CIFAR_CLASSES}.
Architecture (PyTorch, fixed): conv1 = Conv2d(3,32,5,padding=2) -> ReLU -> MaxPool2d(2) -> conv2 = Conv2d(32,64,3,padding=1) -> ReLU
-> AdaptiveAvgPool2d(4) -> flatten (1024 = 64 channels x 4 x 4, channel-major) -> fc = Linear(1024,10).
Design meaningful filters from first principles (e.g. oriented edge/Gabor, color-opponent, center-surround, blob detectors in conv1;
combinations/texture detectors in conv2) and a classifier head encoding your prior knowledge of which colors, textures and spatial
layouts indicate each class (e.g. blue/uniform top regions for airplane/ship, green/brown textures for frog/deer, etc.).
Parameter names and shapes: {json.dumps({k: list(v) for k, v in CIFAR_SHAPES.items()})}
{FORMAT_RULES}"""

@torch.no_grad()
def eval_cifar(m, X, Y, noise=0.0, bs=1000):
    m.eval().to(DEVICE); c = 0; g = torch.Generator().manual_seed(0)
    for i in range(0, len(Y), bs):
        x = X[i:i+bs]
        if noise > 0: x = x + noise * torch.randn(x.shape, generator=g)
        lg = torch.nan_to_num(m(x.to(DEVICE)).float()); c += (lg.argmax(-1).cpu() == Y[i:i+bs]).sum().item()
    return c / len(Y)

def cifar_metrics(m, Xte, Yte):
    r = {"all_acc": eval_cifar(m, Xte, Yte)}
    r["test_acc"] = r["all_acc"]; r["above_chance_pts"] = r["all_acc"] - CIFAR_CHANCE
    for nz in NOISES: r[f"acc_noise{nz}"] = eval_cifar(m, Xte, Yte, nz)
    r.update(structure("cifar", get_weights(m))); return r

def train_A_cifar(Xtr, Ytr, Xval, Yval, opt_name, lr, seed, deadline):
    set_seed(seed); m = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr) if opt_name == "adam" else torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed); comp = 0.0; rows = []
    for step in range(1, max(CIFAR_STEPS) + 1):
        t0 = time.perf_counter(); idx = torch.randint(0, len(Ytr), (CIFAR_BATCH,), generator=g)
        x, y = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
        if torch.rand(1, generator=g).item() < 0.5: x = x.flip(3)
        m.train(); loss = F.cross_entropy(m(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if DEVICE == "cuda": torch.cuda.synchronize()
        comp += time.perf_counter() - t0
        if step in CIFAR_STEPS:
            rows.append({"task": "cifar", "opt": opt_name, "lr": lr, "seed": seed, "step": step, "train_loss": loss.item(),
                         "val_acc": eval_cifar(m, Xval, Yval), "train_compute_s": comp})
        if time.time() > deadline: break
    return m, rows

def ridge_head(m, Xtr, Ytr, lam=1.0, n=10000):
    """Closed-form ridge readout on frozen (LLM-written) conv features. Uses data, no gradients."""
    with torch.no_grad():
        Fs = torch.cat([m.features(Xtr[i:i+1000].to(DEVICE)).cpu() for i in range(0, min(n, len(Ytr)), 1000)]).double()
        Y1 = F.one_hot(Ytr[:len(Fs)], 10).double() * 2 - 1
        mu, sd = Fs.mean(0), Fs.std(0) + 1e-6; Z = torch.cat([(Fs - mu) / sd, torch.ones(len(Fs), 1, dtype=torch.float64)], 1)
        Wr = torch.linalg.solve(Z.T @ Z + lam * len(Fs) * torch.eye(Z.shape[1], dtype=torch.float64), Z.T @ Y1)   # (1025,10)
        Wf = (Wr[:-1] / sd[:, None]).T; bf = Wr[-1] - (mu / sd) @ Wr[:-1]
        m2 = SmallCNN(); m2.load_state_dict(m.state_dict())
        m2.fc.weight.copy_(Wf.float()); m2.fc.bias.copy_(bf.float())
    return m2.to(DEVICE)

CIFAR_DATA = None
def run_cifar_A():
    global CIFAR_DATA
    Xtr, Ytr, Xte, Yte, src = CIFAR_DATA
    nval = min(5000, len(Ytr) // 10); Xval, Yval, Xt, Yt = Xtr[-nval:], Ytr[-nval:], Xtr[:-nval], Ytr[:-nval]
    deadline = time.time() + TIME_LIMIT["A_cifar"]
    for opt_name, lrs in CIFAR_LR_GRID.items():
        for lr in lrs:
            for seed in SEEDS_A:
                if time.time() > deadline: print("A_cifar time limit"); return
                try:
                    m, rows = train_A_cifar(Xt, Yt, Xval, Yval, opt_name, lr, seed, deadline); A_CURVES.extend(rows); last = rows[-1]
                    r = cifar_metrics(m, Xte, Yte)
                    PER_RUN.append({"task": "cifar", "instance": "A", "synth": opt_name, "prompt": "-", "rep": seed, "lr": lr, "steps": last["step"],
                                    "val_acc": last["val_acc"], "compute_s": last["train_compute_s"], "ok": True, "data": src, **r})
                    print(f"[A cifar {opt_name} lr={lr} s={seed}] val {last['val_acc']:.3f} test {r['test_acc']:.3f}")
                except Exception as e: print("A cifar run failed", repr(e))
                save_all()

if RUN_TASKS.get("cifar"):
    try:
        CIFAR_DATA = load_cifar(); print("CIFAR source:", CIFAR_DATA[4], CIFAR_DATA[0].shape)
        if RUN_A: run_cifar_A()
    except Exception as e: print("CIFAR A section failed:", repr(e)); traceback.print_exc()

def cifar_once(syn, rep, section="B_cifar"):
    Xtr, Ytr, Xte, Yte, src = CIFAR_DATA; t0 = time.perf_counter()
    base = {"task": "cifar", "instance": "B", "synth": label_of(syn), "synth_requested": syn["model_id"], "synth_kind": syn["kind"],
            "prompt": "full", "rep": rep, "data": src}
    try:
        text, meta = call_llm(syn, SYSTEM_PROMPT, CIFAR_PROMPT, f"B_cifar_r{rep}", rep=rep, section=section, max_tokens=CLAUDE_MAX_TOKENS["cifar"])
    except BudgetExceeded: raise
    except Exception as e:
        PER_RUN.append({**base, "ok": False, "all_acc": 0.0, "error": repr(e)[:400], "error_kind": "llm_call"}); return
    base["dry_run"] = meta.get("reply_source") in ("dry_run_mock",)
    row = {**base, **meta, **process_reply("cifar", text, meta, f"{re.sub('[^A-Za-z0-9_.-]', '_', label_of(syn))}_cifar_r{rep}", CIFAR_SHAPES, cifar=True)}
    m = row.pop("_model", None)
    if m is not None:
        try:
            rr = cifar_metrics(ridge_head(m, Xtr, Ytr), Xte, Yte)
            PER_RUN.append({**row, "instance": "B+ridge", **rr, "synth_wall_s": time.perf_counter() - t0})
        except Exception as e: print("ridge failed", repr(e))
    row["synth_wall_s"] = time.perf_counter() - t0; PER_RUN.append(row); save_all()
    print(f"[B cifar {label_of(syn)} r{rep}] ok={row['ok']} acc={row['all_acc']:.4f} {row.get('error_kind') or ''}")
