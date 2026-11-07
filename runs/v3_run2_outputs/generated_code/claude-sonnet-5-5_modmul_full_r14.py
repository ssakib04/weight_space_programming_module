def build_weights():
    p = 113
    n = 112
    g = 3
    log = {}
    x = 1
    for e in range(n):
        log[x] = e
        x = (x * g) % p

    K = 55   # frequencies k = 1..55
    M = 4    # phases per frequency
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    j = 0
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        for m in range(M):
            phi = math.pi * m / M
            for a in range(1, p):
                la = log[a]
                W1[j, a] = math.cos(w * la + phi)
                W1[j, 113 + a] = math.cos(w * la - phi)
            for c in range(1, p):
                W2[c, j] = math.cos(w * log[c])
            j += 1
    return {"W1": W1, "W2": W2}