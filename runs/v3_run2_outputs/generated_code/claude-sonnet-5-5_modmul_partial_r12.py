def build_weights():
    p = 113
    n = p - 1
    # find primitive root
    def order_ok(g):
        x = 1
        for e in range(1, n + 1):
            x = x * g % p
            if x == 1:
                return e == n
        return False
    g = next(g for g in range(2, p) if order_ok(g))
    log = {}
    x = 1
    for k in range(n):
        log[x] = k
        x = x * g % p

    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    nf = 32
    for f in range(1, nf + 1):
        w = 2 * math.pi * f / n
        base = (f - 1) * 8
        # (X, Y, sign, coefficient type)
        specs = [
            ("c", "c", +1, "c", +1), ("c", "c", -1, "c", -1),
            ("s", "s", +1, "c", -1), ("s", "s", -1, "c", +1),
            ("s", "c", +1, "s", +1), ("s", "c", -1, "s", -1),
            ("c", "s", +1, "s", +1), ("c", "s", -1, "s", -1),
        ]
        for t, (X, Y, sg, ct, cs) in enumerate(specs):
            row = base + t
            for a in range(1, p):
                la = log[a]
                va = math.cos(w * la) if X == "c" else math.sin(w * la)
                vb = math.cos(w * la) if Y == "c" else math.sin(w * la)
                W1[row, a] = va
                W1[row, p + a] = sg * vb
            for k in range(n):
                c = pow(g, k, p)
                val = math.cos(w * k) if ct == "c" else math.sin(w * k)
                W2[c, row] = cs * val
    return {"W1": W1, "W2": W2}