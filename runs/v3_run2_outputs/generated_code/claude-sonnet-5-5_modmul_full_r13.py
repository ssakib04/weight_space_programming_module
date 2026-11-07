def build_weights():
    p = 113
    n = 112
    g = 3
    # discrete log table
    log = np.zeros(p, dtype=int)
    x = 1
    for e in range(n):
        log[x] = e
        x = (x * g) % p

    K = 32  # frequencies k = 1..32, 8 neurons each = 256
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))

    idx = np.arange(1, p)
    la = log[idx].astype(float)

    cosf = np.cos
    sinf = np.sin
    # (fn on a, fn on b, readout fn, coefficient)
    # cos a cos b - sin a sin b = cos(S);  sin a cos b + cos a sin b = sin(S)
    # cos(S - C) = cos S cos C + sin S sin C
    types = [
        (cosf, cosf, cosf, 0.25),
        (sinf, sinf, cosf, -0.25),
        (sinf, cosf, sinf, 0.25),
        (cosf, sinf, sinf, 0.25),
    ]

    for k in range(1, K + 1):
        th = 2 * math.pi * k / n
        base = 8 * (k - 1)
        for t, (fa, fb, fr, coef) in enumerate(types):
            for si, s in enumerate((1.0, -1.0)):
                j = base + 2 * t + si
                W1[j, idx] = fa(th * la)
                W1[j, p + idx] = s * fb(th * la)
                # act = (u + s v)^2 ; act(+) - act(-) = 4 u v
                W2[idx, j] = s * coef * fr(th * la)

    return {"W1": W1, "W2": W2}