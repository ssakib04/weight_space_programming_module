def build_weights():
    p = 113
    n = 112
    # find a primitive root (3 expected)
    def is_gen(g):
        x = 1
        seen = set()
        for _ in range(n):
            x = (x * g) % p
            seen.add(x)
        return len(seen) == n
    g = 3 if is_gen(3) else next(t for t in range(2, p) if is_gen(t))
    log = np.zeros(p, dtype=np.int64)
    x = 1
    for e in range(n):
        log[x] = e
        x = (x * g) % p

    K = 16   # frequencies k=1..K
    N = 8    # phases per frequency (8*16 = 128 neurons)
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    idx = 0
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        for j in range(N):
            phi = 2 * math.pi * j / N
            psi = 2 * phi
            theta = phi + psi
            for a in range(1, p):
                W1[idx, a] = math.cos(w * log[a] + phi)
                W1[idx, 113 + a] = math.cos(w * log[a] + psi)
                W2[a, idx] = math.cos(w * log[a] + theta)
            idx += 1
    return {"W1": W1, "W2": W2}