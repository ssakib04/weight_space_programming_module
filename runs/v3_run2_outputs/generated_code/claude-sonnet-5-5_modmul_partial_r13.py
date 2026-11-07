def build_weights():
    p = 113
    n = p - 1
    # find a primitive root
    def order_ok(g):
        x = 1
        for e in range(1, n + 1):
            x = (x * g) % p
            if x == 1:
                return e == n
        return False
    g = next(g for g in range(2, p) if order_ok(g))
    dlog = {}
    x = 1
    for e in range(n):
        dlog[x] = e
        x = (x * g) % p

    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    K = 32
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        base = 8 * (k - 1)
        # unit types: (fx, fy, sign, output kind, output sign)
        # fx applied to a (index i), fy applied to b (index j)
        C = lambda t: math.cos(w * t)
        S = lambda t: math.sin(w * t)
        types = [
            (C, C, 'c', 1.0),   # CC -> +cos
            (S, S, 'c', -1.0),  # SS -> -cos
            (S, C, 's', 1.0),   # SC -> +sin
            (C, S, 's', 1.0),   # CS -> +sin
        ]
        for t, (fx, fy, kind, osign) in enumerate(types):
            for sgn in (1.0, -1.0):
                row = base + 2 * t + (0 if sgn > 0 else 1)
                for a in range(1, p):
                    W1[row, a] = fx(dlog[a])
                    W1[row, p + a] = sgn * fy(dlog[a])
                for c in range(1, p):
                    l = dlog[c]
                    val = C(l) if kind == 'c' else S(l)
                    W2[c, row] = sgn * osign * val / 4.0
    return {"W1": W1, "W2": W2}