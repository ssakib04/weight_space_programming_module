def build_weights():
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]
    p = 113
    val = np.array(VAL, dtype=np.float64)          # val[s] = integer encoded by symbol s
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    nfreq = 42                                     # 6 neurons per frequency -> 252 <= 256
    for ki in range(nfreq):
        k = ki + 1
        w = 2.0 * math.pi * k / p
        for j in range(3):
            theta = math.pi * j / 3.0              # 2*theta = 2*pi*j/3
            fa = np.cos(w * val + theta)           # cos(w*A + theta)
            g = np.cos(w * val + 2.0 * theta)      # output weights cos(w*c + 2*theta)
            for sgn_idx, sgn in enumerate((1.0, -1.0)):
                n = 6 * ki + 2 * j + sgn_idx
                # h = cos(wA+theta) + sgn*cos(wB+theta)
                W1[n, :113] = fa
                W1[n, 113:] = sgn * fa
                # (x+y)^2 - (x-y)^2 = 4xy = 2[cos(w(A+B)+2theta) + cos(w(A-B))]
                # the cos(w(A-B)) term cancels over the 3 phases; the rest gives
                # (3/2) cos(w(A+B-c)) after the phase sum
                W2[:, n] = sgn * g
    return {"W1": W1, "W2": W2}
