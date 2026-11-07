"""Offline, network-free evaluator for saved square-activation modular-arithmetic weights.
Architecture: x = concat(onehot(a), onehot(b)) (226); h = W1 @ x; act = h*h; logits = W2 @ act.
Uses NumPy only. Never executes model-generated code; it only loads saved .npz arrays."""
import numpy as np
P = 113
VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]
SHAPES = {"W1": (256, 226), "W2": (113, 256)}
THRESH = 0.99

def domain(task):
    if task == "modmul":
        a, b = np.meshgrid(np.arange(1, P), np.arange(1, P), indexing="ij"); a, b = a.ravel(), b.ravel(); y = (a * b) % P
    elif task == "modadd":
        a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij"); a, b = a.ravel(), b.ravel(); y = (a + b) % P
    elif task == "permadd":
        val = np.array(VAL); inv = np.empty(P, int); inv[val] = np.arange(P)
        a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij"); a, b = a.ravel(), b.ravel(); y = inv[(val[a] + val[b]) % P]
    else:
        raise ValueError(task)
    return a, b, y

def evaluate_arrays(W1, W2, task, dtype=np.float32):
    out = {"task": task, "dtype": np.dtype(dtype).name}
    if W1.shape != SHAPES["W1"] or W2.shape != SHAPES["W2"]:
        return {**out, "ok": False, "success": False, "error": f"shape {W1.shape},{W2.shape}"}
    if not (np.isfinite(W1).all() and np.isfinite(W2).all()):
        return {**out, "ok": False, "success": False, "error": "non-finite weights"}
    a, b, y = domain(task)
    W1 = W1.astype(dtype); W2 = W2.astype(dtype)
    h = W1[:, a].T + W1[:, P + b].T          # equals W1 @ concat(onehot(a), onehot(b))
    logits = (h * h) @ W2.T
    if not np.isfinite(logits).all():
        return {**out, "ok": False, "success": False, "error": "non-finite logits"}
    pred = logits.argmax(1); correct = int((pred == y).sum()); n = len(y)
    return {**out, "ok": True, "correct": correct, "total": n, "accuracy": correct / n, "success": correct / n >= THRESH}

def evaluate_npz(path, task, dtype=np.float32):
    z = np.load(path)
    return evaluate_arrays(np.asarray(z["W1"]), np.asarray(z["W2"]), task, dtype)
