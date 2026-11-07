def build_weights():
    p = 113
    n = 112
    g = 3
    # discrete log table
    lg = np.zeros(p, dtype=int)
    x = 1
    for e in range(n):
        lg[x] = e
        x = (x * g) % p
    w = 2 * math.pi / n
    K = 42   # frequencies 1..42
    M = 3    # phases per frequency
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    j = 0
    for k in range(1, K + 1):
        for m in range(M):
            phi = math.pi * m / M
            for s in (1.0, -1.0):
                for a in range(1, p):
                    W1[j, a] = math.cos(w * k * lg[a] + phi)
                    W1[j, p + a] = s * math.cos(w * k * lg[a] + phi)
                for c in range(1, p):
                    W2[c, j] = s * math.cos(w * k * lg[c] + 2 * phi)
                j += 1
    return {"W1": W1, "W2": W2}