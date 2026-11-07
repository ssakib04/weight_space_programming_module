def build_weights():
    VAL = np.array([
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39,
        75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96,
        60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93,
        14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65,
        69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47,
        15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63,
        48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32,
        92, 80, 35, 22, 79, 59, 84
    ], dtype=np.float64)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    base = 2.0 * math.pi / 113.0

    for k in range(1, 57):
        angle = base * k * VAL
        co = np.cos(angle)
        si = np.sin(angle)
        r = 4 * (k - 1)

        W1[r, :113] = co
        W1[r, 113:] = co

        W1[r + 1, :113] = si
        W1[r + 1, 113:] = -si

        W1[r + 2, :113] = co
        W1[r + 2, 113:] = si

        W1[r + 3, :113] = si
        W1[r + 3, 113:] = co

        target_angle = base * k * VAL
        target_co = 0.5 * np.cos(target_angle)
        target_si = 0.5 * np.sin(target_angle)

        W2[:, r] = target_co
        W2[:, r + 1] = target_co
        W2[:, r + 2] = target_si
        W2[:, r + 3] = target_si

    W1[224, :] = 0.5
    for c in range(113):
        target_angles = base * np.arange(1, 57) * VAL[c]
        W2[c, 224] = -np.sum(np.cos(target_angles) + np.sin(target_angles))

    return {"W1": W1, "W2": W2}
