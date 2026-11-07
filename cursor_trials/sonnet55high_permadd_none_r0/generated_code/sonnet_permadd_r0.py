def build_weights():
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]
    p = 113
    K = 42          # frequencies, 6 hidden units each -> 252 <= 256
    V = np.array(VAL, dtype=np.float64)
    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    row = 0
    for k in range(1, K + 1):
        theta = 2.0 * math.pi * k / p
        for j in range(3):
            alpha = math.pi * j / 3.0      # 2*alpha = 2*pi*j/3
            u = np.cos(theta * V + alpha)
            wc = np.cos(2.0 * alpha + theta * V)   # output weight per symbol c
            # unit (u(a) + u(b))^2 and (u(a) - u(b))^2
            W1[row, :p] = u
            W1[row, p:] = u
            W2[:, row] = wc
            row += 1
            W1[row, :p] = u
            W1[row, p:] = -u
            W2[:, row] = -wc
            row += 1
    return {"W1": W1, "W2": W2}
