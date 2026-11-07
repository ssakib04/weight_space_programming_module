def build_weights():
    p = 113
    n = p - 1
    # find primitive root
    def is_prim(g):
        x = 1
        seen = set()
        for _ in range(n):
            x = (x * g) % p
            seen.add(x)
        return len(seen) == n
    g = next(t for t in range(2, p) if is_prim(t))
    log = np.zeros(p, dtype=int)
    x = 1
    for e in range(n):
        log[x] = e
        x = (x * g) % p

    K = 32
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        base = 8 * (k - 1)
        for v in range(1, p):
            i = log[v]
            ca, sa = math.cos(w * i), math.sin(w * i)
            # a-side columns: v ; b-side columns: p+v
            # n1: cos x + cos y
            W1[base + 0, v] = ca
            W1[base + 0, p + v] = ca
            # n2: sin x + sin y
            W1[base + 1, v] = sa
            W1[base + 1, p + v] = sa
            # n3 (A): -sin x + cos y
            W1[base + 2, v] = -sa
            W1[base + 2, p + v] = ca
            # n4 (B): cos x + sin y
            W1[base + 3, v] = ca
            W1[base + 3, p + v] = sa
            # unary a-only
            W1[base + 4, v] = ca
            W1[base + 5, v] = sa
            # unary b-only
            W1[base + 6, p + v] = ca
            W1[base + 7, p + v] = sa
        for c in range(1, p):
            l = log[c]
            C, S = math.cos(w * l), math.sin(w * l)
            W2[c, base + 0] = C
            W2[c, base + 1] = -C
            W2[c, base + 2] = -S
            W2[c, base + 3] = S
            W2[c, base + 4] = -(C + S)
            W2[c, base + 5] = (C + S)
            W2[c, base + 6] = -(C - S)
            W2[c, base + 7] = (C - S)
    return {"W1": W1, "W2": W2}