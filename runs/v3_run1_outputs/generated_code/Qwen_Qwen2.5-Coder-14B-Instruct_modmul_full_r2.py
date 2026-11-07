def build_weights():
    import numpy as np
    import math

    p = 113
    g = 3
    log_map = np.zeros(p)
    for i in range(1, p):
        log_map[i] = next(j for j in range(p-1) if pow(g, j, p) == i)

    k_values = np.arange(256)
    phase = np.random.rand() * 2 * math.pi
    phase_prime = np.random.rand() * 2 * math.pi

    W1 = np.zeros((256, 226))
    for i in range(1, p):
        for j in range(1, p):
            for k in k_values:
                W1[k, i-1] += np.cos(2 * math.pi * k * log_map[i] / (p-1) + phase)
                W1[k, 112+j-1] += np.cos(2 * math.pi * k * log_map[j] / (p-1) + phase)
            W1[:, i-1] /= np.sqrt(226)
            W1[:, 112+j-1] /= np.sqrt(226)

    W2 = np.zeros((113, 256))
    for c in range(1, p):
        for k in k_values:
            W2[c, k] = np.cos(2 * math.pi * k * log_map[c] / (p-1) + phase + phase_prime)
        W2[c, :] /= np.sqrt(256)

    return {"W1": W1, "W2": W2}