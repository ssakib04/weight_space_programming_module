def build_weights():
    VAL = [
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70,
        8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58,
        91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106,
        86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50,
        12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36,
        29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92,
        80, 35, 22, 79, 59, 84
    ]
    n = 113
    val_arr = np.array(VAL, dtype=np.float64)
    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    omega = 2.0 * math.pi / n
    for k in range(1, 57):
        angles = omega * k * val_arr
        c = np.cos(angles)
        s = np.sin(angles)

        idx = 4 * (k - 1)
        # h_{k,0}^2 + h_{k,1}^2 = 2 + 2*cos(omega * k * (VAL[a] + VAL[b]))
        W1[idx + 0, :n] = c
        W1[idx + 0, n:] = c
        W1[idx + 1, :n] = s
        W1[idx + 1, n:] = -s

        # h_{k,2}^2 + h_{k,3}^2 = 2 + 2*sin(omega * k * (VAL[a] + VAL[b]))
        W1[idx + 2, :n] = c
        W1[idx + 2, n:] = s
        W1[idx + 3, :n] = s
        W1[idx + 3, n:] = c

        W2[:, idx + 0] = c
        W2[:, idx + 1] = c
        W2[:, idx + 2] = s
        W2[:, idx + 3] = s

        # Subtract the constant offset 2*(c + s) using neuron 224 (which has act = 1)
        W2[:, 224] -= 2.0 * (c + s)

    W1[224, :n] = 1.0

    return {"W1": W1, "W2": W2}
