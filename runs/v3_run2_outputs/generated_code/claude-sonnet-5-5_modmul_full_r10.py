def build_weights():
    p = 113
    n = 112
    g = 3
    log = {}
    for i in range(n):
        log[pow(g, i, p)] = i
    K = 42   # frequencies k=1..42, 6 neurons each = 252 <= 256
    M = 3    # phases per frequency
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    j = 0
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        for m in range(M):
            phi = math.pi * m / M
            for s in (1.0, -1.0):
                for a in range(1, p):
                    la = log[a]
                    W1[j, a] = math.cos(w * la + phi)
                    W1[j, 113 + a] = s * math.cos(w * la + phi)
                for c in range(1, p):
                    W2[c, j] = s * math.cos(w * log[c] + 2 * phi)
                j += 1
    return {"W1": W1, "W2": W2}