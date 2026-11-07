def build_weights():
    p, n = 113, 112
    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    primitive = next(
        g for g in range(2, p)
        if pow(g, n // 2, p) != 1 and pow(g, n // 7, p) != 1
    )

    discrete_log = np.zeros(p, dtype=np.int64)
    value = 1
    for exponent in range(n):
        discrete_log[value] = exponent
        value = (value * primitive) % p

    angles = (2.0 * math.pi / n) * discrete_log
    cos_phase = np.cos(angles)
    sin_phase = np.sin(angles)
    cos_phase[0] = sin_phase[0] = 0.0

    terms = (
        (cos_phase, cos_phase,  cos_phase),
        (sin_phase, sin_phase, -cos_phase),
        (sin_phase, cos_phase,  sin_phase),
        (cos_phase, sin_phase,  sin_phase),
    )

    for term_index, (left, right, output) in enumerate(terms):
        plus_row = 2 * term_index
        minus_row = plus_row + 1

        W1[plus_row, :113] = left
        W1[plus_row, 113:] = right
        W1[minus_row, :113] = left
        W1[minus_row, 113:] = -right

        W2[1:, plus_row] = output[1:] / 4.0
        W2[1:, minus_row] = -output[1:] / 4.0

    return {"W1": W1, "W2": W2}
