def build_weights():
    p = 113
    n = p - 1  # 112, order of the multiplicative group

    # find a primitive root g mod p
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
    ms = np.array([log[c] for c in range(1, p)])  # log of c for c=1..112

    idx = 0
    for k in range(1, 33):
        th = 2 * np.pi * k / n
        for phi, use_cos_out in ((0.0, True), (-np.pi / 4, False)):
            # four neurons: (cos, +), (sin, +), (cos, -), (sin, -)
            # output weights: +1, -1, -1, +1 -> 4*cos(th*(i+j)+2phi)
            specs = [("c", 1, 1.0), ("s", 1, -1.0), ("c", -1, -1.0), ("s", -1, 1.0)]
            for kind, sig, w in specs:
                for a in range(1, p):
                    al = th * log[a] + phi
                    v = np.cos(al) if kind == "c" else np.sin(al)
                    W1[idx, a] = v
                    W1[idx, p + a] = sig * v
                # output target: cos(th*s)cos(th*m)+sin(th*s)sin(th*m)
                if use_cos_out:
                    tgt = np.cos(th * ms)
                else:
                    tgt = np.sin(th * ms)
                W2[1:, idx] = w * tgt
                idx += 1
    return {"W1": W1, "W2": W2}