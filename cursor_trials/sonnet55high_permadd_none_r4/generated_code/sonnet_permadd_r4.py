def build_weights():
    VAL = np.array([34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84], dtype=np.float64)
    P = 113
    W1 = np.zeros((256, 2 * P))
    W2 = np.zeros((P, 256))
    for k in range(1, 33):
        th = 2.0 * math.pi * k / P
        cv = np.cos(th * VAL)   # cos(theta * value) per symbol
        sv = np.sin(th * VAL)
        # (fa, fb, sign on fb, output type, output sign)
        # Pairs of squares give (f+g)^2-(f-g)^2 = 4 f g, so there is no leftover junk.
        # Target: cos(th*(a+b-c)) = cos(phi_c)*(ca*cb - sa*sb) + sin(phi_c)*(sa*cb + ca*sb)
        specs = [
            (cv, cv, +1, 'c', +1),
            (cv, cv, -1, 'c', -1),
            (sv, sv, +1, 'c', -1),
            (sv, sv, -1, 'c', +1),
            (sv, cv, +1, 's', +1),
            (sv, cv, -1, 's', -1),
            (cv, sv, +1, 's', +1),
            (cv, sv, -1, 's', -1),
        ]
        base = 8 * (k - 1)
        for j, (fa, fb, sg, typ, osg) in enumerate(specs):
            r = base + j
            W1[r, :P] = fa
            W1[r, P:] = sg * fb
            out = cv if typ == 'c' else sv   # cos(th*VAL[c]) or sin(th*VAL[c]) per output symbol
            W2[:, r] = osg * out / 4.0
    return {"W1": W1, "W2": W2}
