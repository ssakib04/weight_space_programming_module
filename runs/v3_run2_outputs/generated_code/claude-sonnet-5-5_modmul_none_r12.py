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
    dlog = {}
    x = 1
    for e in range(n):
        dlog[x] = e
        x = (x * g) % p

    th = 2 * math.pi / n
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    const_w = np.zeros(p)

    idx = 0
    for k in range(1, 57):
        # pair C: s=t=0 ; pair S: s=pi/2, t=0
        for pair in range(2):
            na, nb = idx, idx + 1
            idx += 2
            for a in range(1, p):
                ang = k * th * dlog[a]
                if pair == 0:
                    fa, fb = math.cos(ang), math.sin(ang)
                else:
                    fa, fb = math.sin(ang), -math.cos(ang)
                W1[na, a] = fa
                W1[nb, a] = fb
            for b in range(1, p):
                ang = k * th * dlog[b]
                if pair == 0:
                    ga, gb = math.cos(ang), -math.sin(ang)
                else:
                    ga, gb = math.cos(ang), -math.sin(ang)
                W1[na, p + b] = ga
                W1[nb, p + b] = gb
            for c in range(1, p):
                l = k * th * dlog[c]
                w = math.cos(l) if pair == 0 else math.sin(l)
                W2[c, na] = w
                W2[c, nb] = w
                const_w[c] += 2 * 2 * 0 + 2 * w  # constant 2 from each pair: 2*w (sum of both neurons' consts is 2)
    # constant neuron: h = 2, act = 4
    W1[idx, :] = 1.0
    for c in range(1, p):
        W2[c, idx] = -const_w[c] / 4.0
    W2[0, idx] = -100.0
    return {"W1": W1, "W2": W2}