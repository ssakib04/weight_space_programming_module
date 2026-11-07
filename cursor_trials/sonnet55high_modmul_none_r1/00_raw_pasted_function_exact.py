def build_weights():
    p = 113
    n = p - 1  # 112, order of the multiplicative group

    # Find a primitive root g of p (order 112 = 2^4 * 7)
    g = 2
    while pow(g, n // 2, p) == 1 or pow(g, n // 7, p) == 1:
        g += 1

    # Discrete log table: log[a] = i such that g^i = a (mod p), for a in 1..112
    log = np.zeros(p, dtype=np.int64)
    v = 1
    for i in range(n):
        log[v] = i
        v = (v * g) % p

    nfreq = 32            # 8 neurons per frequency -> 256 hidden units
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))

    idx = np.arange(1, p)            # valid inputs 1..112 (0 never appears)
    t = log[idx].astype(np.float64)  # discrete logs

    # (function on a, function on b, sign) for the 8 neurons per frequency.
    # Each neuron computes (f(a) + sign*g(b))^2; differences of such squares
    # give 4*f(a)*g(b) and the single-variable squared terms cancel exactly.
    #  n0..n3 -> cos(i)cos(j) and sin(i)sin(j)  => cos(w(i+j))
    #  n4..n7 -> sin(i)cos(j) and cos(i)sin(j)  => sin(w(i+j))
    specs = [("c", "c", +1), ("c", "c", -1),
             ("s", "s", +1), ("s", "s", -1),
             ("s", "c", +1), ("s", "c", -1),
             ("c", "s", +1), ("c", "s", -1)]

    def fn(name, ang):
        return np.cos(ang) if name == "c" else np.sin(ang)

    coef_cos = [1, -1, -1, 1, 0, 0, 0, 0]
    coef_sin = [0, 0, 0, 0, 1, -1, 1, -1]

    for k in range(1, nfreq + 1):
        w = 2.0 * math.pi * k / n
        ang = w * t
        for r, (fa, fb, sg) in enumerate(specs):
            row = 8 * (k - 1) + r
            W1[row, idx] = fn(fa, ang)                 # a part
            W1[row, p + idx] = sg * fn(fb, ang)        # b part
        # Output: logit_c += cos(w*(i+j - log c)) = cos(ws)cos(wt_c) + sin(ws)sin(wt_c)
        for r in range(8):
            col = 8 * (k - 1) + r
            W2[idx, col] = 0.25 * (coef_cos[r] * np.cos(ang) + coef_sin[r] * np.sin(ang))

    return {"W1": W1, "W2": W2}
