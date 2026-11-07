P = 113
H_ALGO = 256
PAR_N, PAR_K, H_PAR = 16, 6, 64
PERM = np.random.default_rng(4321).permutation(113)   # secret relabelling for the recall-resistant task
PAR_SUBSET = sorted(np.random.default_rng(1234).choice(PAR_N, PAR_K, replace=False).tolist())

def primitive_root(p):
    fac = [q for q in range(2, p) if (p - 1) % q == 0 and all(q % r for r in range(2, int(q**.5) + 1))]
    return next(g for g in range(2, p) if all(pow(g, (p - 1) // q, p) != 1 for q in fac))
G_ROOT = primitive_root(P)

class MLP(nn.Module):
    """Shared architecture for A/B/C. Params: W1 (H,in), [b1 (H)], W2 (out,H), [b2 (out)]."""
    def __init__(self, d_in, H, d_out, act="quad", bias=False):
        super().__init__()
        self.act_name, self.bias = act, bias
        self.W1 = nn.Parameter(torch.randn(H, d_in) / math.sqrt(d_in))
        self.W2 = nn.Parameter(torch.randn(d_out, H) / math.sqrt(H))
        if bias: self.b1 = nn.Parameter(torch.zeros(H)); self.b2 = nn.Parameter(torch.zeros(d_out))
    def forward(self, x):
        h = x @ self.W1.T + (self.b1 if self.bias else 0)
        h = h * h if self.act_name == "quad" else F.relu(h)
        return h @ self.W2.T + (self.b2 if self.bias else 0)

def onehot2(a, b, p=P): return torch.cat([F.one_hot(a, p), F.one_hot(b, p)], -1).float()

def make_task(name):
    if name == "permadd":
        a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij"); a, b = a.ravel(), b.ravel()
        inv = np.argsort(PERM); y = inv[(PERM[a] + PERM[b]) % P]
        X = onehot2(torch.tensor(a), torch.tensor(b)); Y = torch.tensor(y)
        shapes = {"W1": (H_ALGO, 2 * P), "W2": (P, H_ALGO)}
        return dict(name=name, X=X, Y=Y, shapes=shapes, n_cls=P, make_model=lambda: MLP(2 * P, H_ALGO, P, "quad", False))
    if name in ("modadd", "modmul"):
        lo = 0 if name == "modadd" else 1
        a, b = np.meshgrid(np.arange(lo, P), np.arange(lo, P), indexing="ij"); a, b = a.ravel(), b.ravel()
        y = (a + b) % P if name == "modadd" else (a * b) % P
        X = onehot2(torch.tensor(a), torch.tensor(b)); Y = torch.tensor(y)
        shapes = {"W1": (H_ALGO, 2 * P), "W2": (P, H_ALGO)}
        return dict(name=name, X=X, Y=Y, shapes=shapes, n_cls=P,
                    make_model=lambda: MLP(2 * P, H_ALGO, P, "quad", False))
    if name == "parity":
        idx = np.arange(2 ** PAR_N); bits = ((idx[:, None] >> np.arange(PAR_N)) & 1).astype(np.float32)
        y = bits[:, PAR_SUBSET].sum(1).astype(int) % 2
        shapes = {"W1": (H_PAR, PAR_N), "b1": (H_PAR,), "W2": (2, H_PAR), "b2": (2,)}
        return dict(name=name, X=torch.tensor(bits), Y=torch.tensor(y), shapes=shapes, n_cls=2,
                    make_model=lambda: MLP(PAR_N, H_PAR, 2, "relu", True))
    raise ValueError(name)

def splits(n, seed):
    perm = np.random.default_rng(seed).permutation(n); a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    return torch.tensor(perm[:a]), torch.tensor(perm[a:b]), torch.tensor(perm[b:])

# ---- Oracle constructors, kept as SOURCE strings so the mock synthesizer can return them as "LLM code" ----
ORACLE_SRC = {}
ORACLE_SRC["modadd"] = r"""
def build_weights():
    p, H, M = 113, 256, 3
    K = H // (M*M); W1 = np.zeros((H, 2*p)); W2 = np.zeros((p, H)); x = np.arange(p); h = 0
    for k in range(1, K+1):
        w = 2*np.pi*k/p
        for i in range(M):
            for j in range(M):
                phi, psi = 2*np.pi*i/M, 2*np.pi*j/M
                W1[h, :p] = np.cos(w*x+phi); W1[h, p:] = np.cos(w*x+psi); W2[:, h] = np.cos(w*x+phi+psi); h += 1
    return {"W1": W1, "W2": W2 / (K*M*M/2)}
"""
ORACLE_SRC["modmul"] = r"""
def build_weights():
    p, H, M, g = 113, 256, 3, GROOT
    n = p - 1; log = np.zeros(p, dtype=int); v = 1
    for e in range(n): log[v] = e; v = (v*g) % p
    K = H // (M*M); W1 = np.zeros((H, 2*p)); W2 = np.zeros((p, H)); L = log[1:]; h = 0
    for k in range(1, K+1):
        w = 2*np.pi*k/n
        for i in range(M):
            for j in range(M):
                phi, psi = 2*np.pi*i/M, 2*np.pi*j/M
                W1[h, 1:p] = np.cos(w*L+phi); W1[h, p+1:] = np.cos(w*L+psi); W2[1:, h] = np.cos(w*L+phi+psi); h += 1
    W2[0, :] = 0.0
    return {"W1": W1, "W2": W2 / (K*M*M/2)}
""".replace("GROOT", str(G_ROOT))
ORACLE_SRC["permadd"] = r"""
def build_weights():
    p, H, M = 113, 256, 3
    V = np.array(PERMLIST)
    K = H // (M*M); W1 = np.zeros((H, 2*p)); W2 = np.zeros((p, H)); h = 0
    for k in range(1, K+1):
        w = 2*np.pi*k/p
        for i in range(M):
            for j in range(M):
                phi, psi = 2*np.pi*i/M, 2*np.pi*j/M
                W1[h, :p] = np.cos(w*V+phi); W1[h, p:] = np.cos(w*V+psi); W2[:, h] = np.cos(w*V+phi+psi); h += 1
    return {"W1": W1, "W2": W2 / (K*M*M/2)}
""".replace("PERMLIST", str(PERM.tolist()))
ORACLE_SRC["parity"] = r"""
def build_weights():
    n, H, S = 16, 64, SUBSET
    k = len(S); W1 = np.zeros((H, n)); b1 = np.zeros(H); W2 = np.zeros((2, H)); b2 = np.array([1.0, -1.0])
    for j in range(k+1):
        W1[j, S] = 1.0; b1[j] = -j
        c = 2.0 if j == 0 else 4.0*(-1)**j
        W2[1, j] = c; W2[0, j] = -c
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
""".replace("SUBSET", str(PAR_SUBSET))

SYSTEM_PROMPT = ("You are an expert in mechanistic interpretability who writes neural network weights directly, "
                 "without any training, data fitting or gradients.")
FORMAT_RULES = """Return ONLY one python code block defining `def build_weights():` (no arguments) that uses only `np` (numpy, already
imported) and `math`; no other imports, no file or network access, no randomness unless seeded. It must return a dict mapping
each parameter name to a numpy array of exactly the shape listed. Do not print large arrays; compute them in code."""

def task_prompt(task, level):
    """Prompts for modadd/modmul/parity are byte-identical to v2 so v2 replies are valid cache entries."""
    if task == "modadd":
        spec = f"""Construct weights for a network computing (a+b) mod p, p={P}, for ALL p^2 pairs a,b in 0..{P-1}.
Architecture (fixed): x = concat(onehot(a), onehot(b)) (dim {2*P}); h = W1 @ x, W1 shape ({H_ALGO}, {2*P}), no bias;
act = h*h (elementwise square); logits = W2 @ act, W2 shape ({P}, {H_ALGO}), no bias. argmax(logits) must equal (a+b) mod p."""
        hint = {"full": """Hint: Fourier features. With w_k = 2*pi*k/p, embed a and b as cos(w_k a + phase), cos(w_k b + phase'); squaring
yields cross terms cos(w_k(a+b)+phase+phase') via cos x cos y = [cos(x+y)+cos(x-y)]/2. Read out with W2[c,h] = cos(w_k c + phase+phase')
so logit_c ~ sum_k cos(w_k(a+b-c)). Use a grid of phases so unwanted terms (cos 2wa, cos 2wb, cos w(a-b), constants) cancel.""",
                "partial": "Hint: a solution exists using periodic (trigonometric) features and trig product identities.",
                "none": ""}[level]
    elif task == "modmul":
        spec = f"""Construct weights for a network computing (a*b) mod p, p={P}, for ALL pairs a,b in 1..{P-1} (0 never appears as input).
Architecture (fixed): x = concat(onehot(a), onehot(b)) with one-hot over 0..{P-1} (dim {2*P}); h = W1 @ x, W1 shape ({H_ALGO}, {2*P}), no bias;
act = h*h; logits = W2 @ act, W2 shape ({P}, {H_ALGO}), no bias, logits indexed by c in 0..{P-1}. argmax(logits) must equal (a*b) mod p."""
        hint = {"full": f"""Hint: the multiplicative group mod p is cyclic of order {P-1} with generator g={G_ROOT}. Let log(a) be the discrete log base g.
Then a*b mod p corresponds to log(a)+log(b) mod {P-1}. Apply the Fourier construction for modular addition on Z_{P-1}: embed
cos(2*pi*k*log(a)/{P-1} + phase), square, read out with cos(2*pi*k*log(c)/{P-1} + phase+phase'), phase grid so unwanted terms cancel; row c=0 can be zero.""",
                "partial": "Hint: use the structure of the multiplicative group mod p (it is cyclic) and periodic features.",
                "none": ""}[level]
    elif task == "permadd":
        spec = f"""Symbols 0..{P-1} encode the elements of Z_{P} through a SECRET relabelling: symbol s stands for the integer VAL[s], where
VAL = {PERM.tolist()}.
Construct weights for a network that, given symbols a and b, outputs the symbol c with VAL[c] = (VAL[a] + VAL[b]) mod {P}, for ALL {P}^2 pairs.
Architecture (fixed): x = concat(onehot(a), onehot(b)) over symbols (dim {2*P}); h = W1 @ x, W1 shape ({H_ALGO}, {2*P}), no bias;
act = h*h (elementwise square); logits = W2 @ act, W2 shape ({P}, {H_ALGO}), no bias, indexed by output symbol c."""
        hint = {"full": """Hint: Fourier features on the relabelled values. With w_k = 2*pi*k/p, embed symbol a as cos(w_k VAL[a] + phase), likewise b;
squaring yields cos(w_k(VAL[a]+VAL[b]) + phase+phase'). Read out with W2[c,h] = cos(w_k VAL[c] + phase+phase'); use a phase grid so unwanted terms cancel.""",
                "partial": "Hint: the relabelled operation is a cyclic group; periodic features of the underlying values work.",
                "none": ""}[level]
    elif task == "parity":
        spec = f"""Construct weights for a network computing the parity (XOR) of the input bits at positions S={PAR_SUBSET} (0-indexed) of a
{PAR_N}-bit input x in {{0,1}}^{PAR_N}, for ALL 2^{PAR_N} inputs; the other bits are distractors.
Architecture (fixed): h = relu(W1 @ x + b1), W1 ({H_PAR}, {PAR_N}), b1 ({H_PAR},); logits = W2 @ h + b2, W2 (2, {H_PAR}), b2 (2,).
Class 1 = odd parity, class 0 = even. argmax(logits) must be correct for every input."""
        hint = {"full": f"""Hint: let s = sum of the bits in S (an integer 0..{PAR_K}). Use units relu(s - j) for j=0..{PAR_K} and choose output weights
so the piecewise-linear function of s equals +1 at odd s and -1 at even s (a zig-zag); logits = [-f, f] plus bias.""",
                "partial": "Hint: the answer only depends on the count of active bits in S; a piecewise-linear function of that count suffices.",
                "none": ""}[level]
    shapes = TASKS[task]["shapes"]
    return f"{spec}\n{hint}\nParameter names and shapes: {json.dumps({k: list(v) for k, v in shapes.items()})}\n{FORMAT_RULES}"

TASKS = {t: make_task(t) for t in ["modadd", "modmul", "parity", "permadd"] if RUN_TASKS.get(t)}
for t, T in TASKS.items(): print(t, "N =", len(T["Y"]), "shapes", T["shapes"])
print("primitive root mod", P, "=", G_ROOT, "| parity subset", PAR_SUBSET)
