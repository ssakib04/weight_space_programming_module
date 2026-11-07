def build_weights():
    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    p, order = 113, 112

    for g in range(2, p):
        if pow(g, order // 2, p) != 1 and pow(g, order // 7, p) != 1:
            generator = g
            break

    discrete_log = np.zeros(p, dtype=np.int64)
    value = 1
    for exponent in range(order):
        discrete_log[value] = exponent
        value = (value * generator) % p

    angles = 2.0 * np.pi * discrete_log / order
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)

    feature_pairs = (
        (cos_values, cos_values),
        (sin_values, sin_values),
        (sin_values, cos_values),
        (cos_values, sin_values),
    )

    for pair_index, (left, right) in enumerate(feature_pairs):
        plus = 2 * pair_index
        minus = plus + 1

        W1[plus, :113] = left
        W1[plus, 113:] = right
        W1[minus, :113] = left
        W1[minus, 113:] = -right

    W1[8, :113] = 1.0
    W2[0, 8] = -2.0

    for c in range(1, p):
        cos_c = cos_values[c]
        sin_c = sin_values[c]
        coefficients = (cos_c, -cos_c, sin_c, sin_c)

        for pair_index, coefficient in enumerate(coefficients):
            W2[c, 2 * pair_index] = coefficient / 4.0
            W2[c, 2 * pair_index + 1] = -coefficient / 4.0

    return {"W1": W1, "W2": W2}
