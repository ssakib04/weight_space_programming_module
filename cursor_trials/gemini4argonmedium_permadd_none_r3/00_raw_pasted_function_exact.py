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
    val_arr = np.array(VAL, dtype=np.float64)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # Constant unit: since input is concat(onehot(a), onehot(b)),
    # setting weights on a-slots to 1 gives h[224] = 1 and act[224] = 1 for all pairs (a, b).
    W1[224, :p] = 1.0

    for k in range(1, 57):
        idx = 4 * (k - 1)
        theta = 2.0 * math.pi * k * val_arr / p
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Unit idx + 0: cos(theta_a) + cos(theta_b)
        W1[idx + 0, :p] = cos_t
        W1[idx + 0, p:] = cos_t

        # Unit idx + 1: sin(theta_a) - sin(theta_b)
        # Note: (idx+0)^2 + (idx+1)^2 = 2 + 2*cos(theta_a + theta_b)
        W1[idx + 1, :p] = sin_t
        W1[idx + 1, p:] = -sin_t

        # Unit idx + 2: cos(theta_a) + sin(theta_b)
        W1[idx + 2, :p] = cos_t
        W1[idx + 2, p:] = sin_t

        # Unit idx + 3: sin(theta_a) + cos(theta_b)
        # Note: (idx+2)^2 + (idx+3)^2 = 2 + 2*sin(theta_a + theta_b)
        W1[idx + 3, :p] = sin_t
        W1[idx + 3, p:] = cos_t

        # Combine with cos(theta_c) and sin(theta_c) to form 2*cos(theta_a + theta_b - theta_c)
        W2[:, idx + 0] = cos_t
        W2[:, idx + 1] = cos_t
        W2[:, idx + 2] = sin_t
        W2[:, idx + 3] = sin_t

        # Cancel the 2*(cos(theta_c) + sin(theta_c)) baseline using the constant unit
        W2[:, 224] -= 2.0 * (cos_t + sin_t)

    return {"W1": W1, "W2": W2}
