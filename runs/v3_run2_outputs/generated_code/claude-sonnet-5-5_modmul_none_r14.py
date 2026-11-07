def build_weights():
    p = 113
    n = p - 1
    # find a primitive root
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

    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    K = 32
    vals = np.arange(1, p)
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        ang = w * log[vals]
        ca = np.cos(ang)
        sa = np.sin(ang)
        base = 8 * (k - 1)
        # (u_a, sign, v_b) per neuron
        specs = [
            (ca, 1, ca),   # 0: (ca+cb)
            (ca, -1, ca),  # 1: (ca-cb)
            (sa, 1, sa),   # 2: (sa+sb)
            (sa, -1, sa),  # 3: (sa-sb)
            (sa, 1, ca),   # 4: (sa+cb)
            (sa, -1, ca),  # 5: (sa-cb)
            (ca, 1, sa),   # 6: (ca+sb)
            (ca, -1, sa),  # 7: (ca-sb)
        ]
        for i, (u, s, v) in enumerate(specs):
            W1[base + i, vals] = u
            W1[base + i, p + vals] = s * v
        # output: cos(A+B)cosC + sin(A+B)sinC
        cc = ca  # same function of c
        sc = sa
        coefs_cos = [1, -1, -1, 1]
        coefs_sin = [1, -1, 1, -1]
        for i in range(4):
            W2[vals, base + i] = 0.25 * coefs_cos[i] * cc
            W2[vals, base + 4 + i] = 0.25 * coefs_sin[i] * sc
    return {"W1": W1, "W2": W2}