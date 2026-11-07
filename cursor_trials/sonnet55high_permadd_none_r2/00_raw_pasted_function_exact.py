def build_weights():
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]
    p = 113
    val = np.array(VAL, dtype=np.float64)
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    K = 40  # frequencies 1..40, 6 neurons each = 240 neurons (rest stay zero)
    n = 0
    for f in range(1, K + 1):
        alpha = 2.0 * math.pi * f * val / p  # angle for each symbol
        for j in range(3):
            phi = math.pi * j / 3.0
            u = np.cos(alpha - phi)
            w = np.cos(2.0 * phi - alpha)  # output weights, indexed by output symbol
            for sign in (1.0, -1.0):
                # h = u[a] + sign*u[b]; (h+)^2 - (h-)^2 = 4 u[a] u[b]
                W1[n, :p] = u
                W1[n, p:] = sign * u
                W2[:, n] = sign * w
                n += 1
    # logit_c ∝ sum_f cos(2*pi*f*(VAL[a]+VAL[b]-VAL[c])/113), uniquely maximal at the correct c
    return {"W1": W1, "W2": W2}
