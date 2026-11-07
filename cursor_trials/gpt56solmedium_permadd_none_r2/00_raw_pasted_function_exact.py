def build_weights():
    vals = np.array([
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39,
        75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96,
        60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93,
        14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65,
        69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47,
        15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63,
        48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32,
        92, 80, 35, 22, 79, 59, 84
    ], dtype=np.float64)

    theta = (2.0 * math.pi / 113.0) * vals
    co = np.cos(theta)
    si = np.sin(theta)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    terms = [
        (co, co,  co),
        (si, si, -co),
        (si, co,  si),
        (co, si,  si),
    ]

    for i, (fa, fb, output_coefficient) in enumerate(terms):
        plus_row = 2 * i
        minus_row = plus_row + 1

        W1[plus_row, :113] = fa
        W1[plus_row, 113:] = fb
        W1[minus_row, :113] = fa
        W1[minus_row, 113:] = -fb

        W2[:, plus_row] = output_coefficient / 4.0
        W2[:, minus_row] = -output_coefficient / 4.0

    return {"W1": W1, "W2": W2}
