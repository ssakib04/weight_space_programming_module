def build_weights():
    p = 113
    n = p - 1
    # find a generator of the multiplicative group
    def order(g):
        x = g
        k = 1
        while x != 1:
            x = x * g % p
            k += 1
        return k
    g = next(t for t in range(2, p) if order(t) == n)
    dlog = {}
    x = 1
    for e in range(n):
        dlog[x] = e
        x = x * g % p

    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    K = 32
    for m in range(K):
        k = m + 1
        w = 2 * math.pi * k / n
        r = 8 * m
        for a in range(1, p):
            i = dlog[a]
            ca, sa = math.cos(w * i), math.sin(w * i)
            # a-part columns: index a ; b-part columns: 113 + b
            # cos-sum neurons
            W1[r, a] = ca
            W1[r + 1, a] = ca
            W1[r + 2, a] = sa
            W1[r + 3, a] = sa
            # sin-sum neurons
            W1[r + 4, a] = sa
            W1[r + 5, a] = sa
            W1[r + 6, a] = ca
            W1[r + 7, a] = ca
            b = a
            cb, sb = ca, sa
            col = 113 + b
            W1[r, col] = cb
            W1[r + 1, col] = -cb
            W1[r + 2, col] = sb
            W1[r + 3, col] = -sb
            W1[r + 4, col] = cb
            W1[r + 5, col] = -cb
            W1[r + 6, col] = sb
            W1[r + 7, col] = -sb
        for c in range(1, p):
            nc = dlog[c]
            cn, sn = math.cos(w * nc), math.sin(w * nc)
            W2[c, r] = cn / 4
            W2[c, r + 1] = -cn / 4
            W2[c, r + 2] = -cn / 4
            W2[c, r + 3] = cn / 4
            W2[c, r + 4] = sn / 4
            W2[c, r + 5] = -sn / 4
            W2[c, r + 6] = sn / 4
            W2[c, r + 7] = -sn / 4
    return {"W1": W1, "W2": W2}