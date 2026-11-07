def build_weights():
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]
    p = 113
    v = np.array(VAL, dtype=np.float64)
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Each unit: h = tA(a) + sign * tB(b); squared. The signed sum of squares gives
    # 4*x*y, which yields cos(S) and sin(S) with S = w*(a+b).
    # (typeA, typeB, sign, output-kind, output-weight)
    spec = [
        # cos(S) = cos a cos b - sin a sin b
        ('c', 'c', +1, 'c', +0.25),
        ('c', 'c', -1, 'c', -0.25),
        ('s', 's', +1, 'c', -0.25),
        ('s', 's', -1, 'c', +0.25),
        # sin(S) = sin a cos b + cos a sin b
        ('s', 'c', +1, 's', +0.25),
        ('s', 'c', -1, 's', -0.25),
        ('c', 's', +1, 's', +0.25),
        ('c', 's', -1, 's', -0.25),
    ]

    u = 0
    for k in range(1, 33):  # 32 frequencies * 8 units = 256 hidden units
        w = 2.0 * math.pi * k / p
        trig = {'c': np.cos(w * v), 's': np.sin(w * v)}
        for (ta, tb, sg, kind, wt) in spec:
            W1[u, :113] = trig[ta]
            W1[u, 113:] = sg * trig[tb]
            # logit[c] += wt * (cos or sin)(w * VAL[c]) * act
            W2[:, u] = wt * trig[kind]
            u += 1

    # logit[c] = sum_k [cos(S_k)cos(C_k) + sin(S_k)sin(C_k)] = sum_k cos(w_k (a+b-c)),
    # uniquely maximal (value 32) at VAL[c] = (VAL[a]+VAL[b]) mod 113.
    return {"W1": W1, "W2": W2}
