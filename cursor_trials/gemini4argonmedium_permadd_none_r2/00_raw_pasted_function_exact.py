def build_weights():
    VAL = [
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45,
        64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54,
        51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74,
        65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46,
        28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19,
        5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84
    ]
    p = 113
    val = np.array(VAL, dtype=np.float64)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # Constant unit h[224] = 1 for any valid input pair (a, b)
    W1[224, :p] = 1.0

    for k in range(1, 57):
        idx = 4 * (k - 1)
        angles = 2.0 * math.pi * k * val / p
        c = np.cos(angles)
        s = np.sin(angles)

        # (cos a + cos b)^2 + (sin a - sin b)^2 = 2 + 2 cos(a + b)
        W1[idx, :p] = c
        W1[idx, p:] = c
        W1[idx + 1, :p] = s
        W1[idx + 1, p:] = -s

        # (sin a + cos b)^2 + (cos a + sin b)^2 = 2 + 2 sin(a + b)
        W1[idx + 2, :p] = s
        W1[idx + 2, p:] = c
        W1[idx + 3, :p] = c
        W1[idx + 3, p:] = s

        W2[:, idx] = c
        W2[:, idx + 1] = c
        W2[:, idx + 2] = s
        W2[:, idx + 3] = s

        # Subtract the 2*cos(c) + 2*sin(c) offset via the constant unit
        W2[:, 224] -= 2.0 * (c + s)

    return {"W1": W1, "W2": W2}
