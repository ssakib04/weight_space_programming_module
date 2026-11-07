def build_weights():
    p = 113
    n = p - 1  # order of the multiplicative group (112)

    # Find a primitive root g mod 113 (order 112: 112 = 2^4 * 7).
    g = None
    for cand in range(2, p):
        if pow(cand, n // 2, p) != 1 and pow(cand, n // 7, p) != 1:
            g = cand
            break

    # Discrete log table: dlog[a] = i with g^i = a (a in 1..112); antilog[i] = g^i.
    dlog = {}
    antilog = [0] * n
    v = 1
    for i in range(n):
        dlog[v] = i
        antilog[i] = v
        v = (v * g) % p

    K = n // 2  # frequencies k = 1..56
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))

    const_idx = 4 * K  # 224: constant neuron (always sees exactly one active a-input)
    W1[const_idx, 0:p] = 1.0
    bias = np.zeros(p)  # per-class constant correction, delivered via the constant neuron

    for k in range(1, K + 1):
        w = 2.0 * math.pi * k / n
        base = 4 * (k - 1)
        for a in range(1, p):
            i = dlog[a]
            ca, sa = math.cos(w * i), math.sin(w * i)
            # P pair: n1 = cos(wi)+cos(wj), n2 = sin(wi)-sin(wj)
            #   n1^2 + n2^2 = 2 + 2cos(w(i+j))
            W1[base + 0, a] = ca
            W1[base + 1, a] = sa
            # Q pair: n3 = sin(wi)+cos(wj), n4 = cos(wi)+sin(wj)
            #   n3^2 + n4^2 = 2 + 2sin(w(i+j))
            W1[base + 2, a] = sa
            W1[base + 3, a] = ca
        for b in range(1, p):
            j = dlog[b]
            cb, sb = math.cos(w * j), math.sin(w * j)
            W1[base + 0, p + b] = cb
            W1[base + 1, p + b] = -sb
            W1[base + 2, p + b] = cb
            W1[base + 3, p + b] = sb

        for m in range(n):
            c = antilog[m]
            cm, sm = math.cos(w * m), math.sin(w * m)
            W2[c, base + 0] = 0.5 * cm
            W2[c, base + 1] = 0.5 * cm
            W2[c, base + 2] = 0.5 * sm
            W2[c, base + 3] = 0.5 * sm
            bias[c] -= (cm + sm)

    # logit[c = g^m] = sum_k cos(w_k (i + j - m)):
    #   = 56 if m == i+j (mod 112), otherwise <= 0.
    W2[:, const_idx] = bias
    W2[0, :] = 0.0  # class 0 never the answer (logit 0 < 56)

    return {"W1": W1, "W2": W2}
