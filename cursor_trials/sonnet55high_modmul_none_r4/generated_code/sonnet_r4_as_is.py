def build_weights():
    p = 113
    n = p - 1  # 112

    # primitive root mod p (found by search, no randomness)
    g = None
    for cand in range(2, p):
        if pow(cand, n // 2, p) != 1 and pow(cand, n // 7, p) != 1:
            g = cand
            break

    # discrete log table: log[g^i mod p] = i; log[0] is unused (0 never appears)
    log = np.zeros(p, dtype=np.int64)
    v = 1
    for i in range(n):
        log[v] = i
        v = (v * g) % p

    K = 32  # frequencies k = 1..32 over Z_112; 32 freqs * 4 products * 2 signs = 256 neurons
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))

    A = np.arange(p)
    la = log[A].astype(float)
    cs = np.arange(1, p)  # c = 1..112
    lc = log[cs].astype(float)

    idx = 0
    for k in range(1, K + 1):
        w = 2.0 * math.pi * k / n
        cx = np.cos(w * la); sx = np.sin(w * la)
        # (u-function, v-function, output coefficient over c)
        types = [
            (cx, cx, np.cos(w * lc)),    # cos x cos y  -> cos(ws) part (+)
            (sx, sx, -np.cos(w * lc)),   # sin x sin y  -> cos(ws) part (-)
            (sx, cx, np.sin(w * lc)),    # sin x cos y  -> sin(ws) part
            (cx, sx, np.sin(w * lc)),    # cos x sin y  -> sin(ws) part
        ]
        for (u, vv, coef) in types:
            for s in (1.0, -1.0):
                # h = u(a) + s*v(b); h^2 difference of the +/- pair gives 4*u(a)*v(b)
                W1[idx, :p] = u
                W1[idx, p:] = s * vv
                W2[cs, idx] = s * coef / 4.0
                idx += 1

    # logit_c = sum_k cos(w_k (log a + log b - log c)), uniquely maximal (=32) at c = a*b mod p.
    # Row c=0 stays zero (logit 0 < 32), so it never wins.
    return {"W1": W1, "W2": W2}