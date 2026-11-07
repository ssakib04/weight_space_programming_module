def build_weights():
    p = 113
    n = p - 1  # 112

    # Find a primitive root g modulo p
    g = 3
    for cand in range(2, p):
        seen = set()
        val = 1
        for _ in range(n):
            seen.add(val)
            val = (val * cand) % p
        if len(seen) == n:
            g = cand
            break

    # Discrete logarithm table: dlog[x] in 0..111 for x in 1..112
    dlog = [0] * p
    val = 1
    for e in range(n):
        dlog[val] = e
        val = (val * g) % p

    W1 = np.zeros((256, 2 * p), dtype=np.float64)
    W2 = np.zeros((p, 256), dtype=np.float64)

    # Neuron 0: constant 1 for all valid inputs a in 1..112
    for a in range(1, p):
        W1[0, a] = 1.0

    # Neurons 1..220: 4 neurons per frequency k in 1..55
    for k in range(1, n // 2):
        idx = 4 * k - 3
        for x in range(1, p):
            angle = 2.0 * math.pi * k * dlog[x] / n
            ca = math.cos(angle)
            sa = math.sin(angle)

            # u1 = cos(alpha_a) + cos(beta_b)
            W1[idx, x] = ca
            W1[idx, p + x] = ca

            # u2 = sin(alpha_a) - sin(beta_b)
            W1[idx + 1, x] = sa
            W1[idx + 1, p + x] = -sa

            # u3 = cos(alpha_a) + sin(beta_b)
            W1[idx + 2, x] = ca
            W1[idx + 2, p + x] = sa

            # u4 = sin(alpha_a) + cos(beta_b)
            W1[idx + 3, x] = sa
            W1[idx + 3, p + x] = ca

    # Neuron 221: Nyquist frequency k = 56
    nyq_idx = 4 * (n // 2) - 3
    for x in range(1, p):
        s = -1.0 if (dlog[x] % 2 == 1) else 1.0
        W1[nyq_idx, x] = s
        W1[nyq_idx, p + x] = s

    # Output weights W2: exact Fourier inversion over Z_112
    for c in range(1, p):
        lc = dlog[c]
        bias_correction = 1.0

        for k in range(1, n // 2):
            idx = 4 * k - 3
            angle = 2.0 * math.pi * k * lc / n
            cc = math.cos(angle)
            sc = math.sin(angle)

            W2[c, idx] = cc
            W2[c, idx + 1] = cc
            W2[c, idx + 2] = sc
            W2[c, idx + 3] = sc

            bias_correction -= 2.0 * (cc + sc)

        sc_nyq = -1.0 if (lc % 2 == 1) else 1.0
        W2[c, nyq_idx] = 0.5 * sc_nyq
        bias_correction -= sc_nyq

        W2[c, 0] = bias_correction

    return {"W1": W1, "W2": W2}
