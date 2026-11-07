def build_weights():
    p = 113
    n = p - 1
    # find primitive root
    def order_ok(g):
        x = 1
        for i in range(1, n + 1):
            x = x * g % p
            if x == 1:
                return i == n
        return False
    g = next(t for t in range(2, p) if order_ok(t))
    log = np.zeros(p, dtype=int)
    x = 1
    for i in range(n):
        log[x] = i
        x = x * g % p

    K = 55
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    vals = np.arange(1, p)
    la = log[vals]
    for k in range(1, K + 1):
        w = 2 * np.pi * k / n
        ca, sa = np.cos(w * la), np.sin(w * la)
        r = 4 * (k - 1)
        # h1 = cosA + cosB
        W1[r, vals] = ca;       W1[r, p + vals] = ca
        # h2 = sinA - sinB
        W1[r + 1, vals] = sa;   W1[r + 1, p + vals] = -sa
        # h3 = cosA + sinB
        W1[r + 2, vals] = ca;   W1[r + 2, p + vals] = sa
        # h4 = sinA + cosB
        W1[r + 3, vals] = sa;   W1[r + 3, p + vals] = ca
        # output weights (c = 1..112)
        cc, sc = np.cos(w * la), np.sin(w * la)
        W2[vals, r] = cc / 2
        W2[vals, r + 1] = cc / 2
        W2[vals, r + 2] = sc / 2
        W2[vals, r + 3] = sc / 2
        # constant neuron compensation
        W2[vals, 4 * K] -= (cc + sc)
    # constant neuron: activation 1 for any a
    W1[4 * K, vals] = 1.0
    return {"W1": W1, "W2": W2}