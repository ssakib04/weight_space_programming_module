def build_weights():
    p = 113
    n = p - 1
    # find a primitive root
    g = None
    for cand in range(2, p):
        if pow(cand, n // 2, p) != 1 and pow(cand, n // 7, p) != 1:
            g = cand
            break
    dlog = {}
    v = 1
    for e in range(n):
        dlog[v] = e
        v = (v * g) % p

    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    for m in range(1, 33):
        w = 2 * math.pi * m / n
        base = (m - 1) * 8
        for x in range(1, p):
            i = dlog[x]
            ci, si = math.cos(w * i), math.sin(w * i)
            # (a-coeff, b-coeff) per neuron, using function of i for a and of j for b
            # neuron: u(a) + s*v(b)
            # 1: c+c, 2: c-c, 3: s+s, 4: s-s, 5: s_a + c_b, 6: s_a - c_b, 7: c_a + s_b, 8: c_a - s_b
            A = [ci, ci, si, si, si, si, ci, ci]
            B = [ci, -ci, si, -si, ci, -ci, si, -si]
            for t in range(8):
                W1[base + t, x] = A[t]
                W1[base + t, p + x] = B[t]
        for c in range(1, p):
            k = dlog[c]
            ck, sk = math.cos(w * k), math.sin(w * k)
            coefs = [ck, -ck, -ck, ck, sk, -sk, sk, -sk]
            for t in range(8):
                W2[c, base + t] = coefs[t] / 4.0
    return {"W1": W1, "W2": W2}