def build_weights():
    val = np.array([
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45,
        64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54,
        51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74,
        65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46,
        28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19,
        5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84
    ], dtype=np.float64)

    n = 113
    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # Constant unit: h[224] = 1 for any one-hot input pair (a, b), so act[224] = 1
    W1[224, :n] = 1.0

    for k in range(1, 57):
        idx = 4 * (k - 1)
        theta = 2.0 * math.pi * k * val / n
        c = np.cos(theta)
        s = np.sin(theta)

        # (cos(theta_a) + cos(theta_b))^2 + (sin(theta_a) - sin(theta_b))^2
        # = 2 + 2 * cos(theta_a + theta_b)
        W1[idx + 0, :n] = c
        W1[idx + 0, n:] = c
        W1[idx + 1, :n] = s
        W1[idx + 1, n:] = -s

        # (cos(theta_a) + sin(theta_b))^2 + (sin(theta_a) + cos(theta_b))^2
        # = 2 + 2 * sin(theta_a + theta_b)
        W1[idx + 2, :n] = c
        W1[idx + 2, n:] = s
        W1[idx + 3, :n] = s
        W1[idx + 3, n:] = c

        # Combine with cos(theta_c) and sin(theta_c) to form 2 * cos(theta_a + theta_b - theta_c)
        W2[:, idx + 0] = c
        W2[:, idx + 1] = c
        W2[:, idx + 2] = s
        W2[:, idx + 3] = s

        # Cancel the 2 * (cos(theta_c) + sin(theta_c)) baseline offset using unit 224
        W2[:, 224] -= 2.0 * (c + s)

    return {"W1": W1, "W2": W2}
