def build_weights():
    p = 113
    n = p - 1

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    g = next(
        candidate
        for candidate in range(2, p)
        if pow(candidate, n // 2, p) != 1
        and pow(candidate, n // 7, p) != 1
    )

    discrete_log = np.zeros(p, dtype=np.int64)
    value = 1
    for exponent in range(n):
        discrete_log[value] = exponent
        value = (value * g) % p

    angles = (2.0 * math.pi / n) * discrete_log
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)

    a_cols = np.arange(1, p)
    b_cols = p + np.arange(1, p)

    pairs = (
        (cos_values, cos_values),
        (sin_values, sin_values),
        (sin_values, cos_values),
        (cos_values, sin_values),
    )

    for pair_index, (left, right) in enumerate(pairs):
        plus_row = 2 * pair_index
        minus_row = plus_row + 1

        W1[plus_row, a_cols] = left[1:]
        W1[plus_row, b_cols] = right[1:]
        W1[minus_row, a_cols] = left[1:]
        W1[minus_row, b_cols] = -right[1:]

    for c in range(1, p):
        theta = angles[c]
        ct = math.cos(theta)
        st = math.sin(theta)

        coefficients = (ct, -ct, st, st)
        for pair_index, coefficient in enumerate(coefficients):
            W2[c, 2 * pair_index] = 0.25 * coefficient
            W2[c, 2 * pair_index + 1] = -0.25 * coefficient

    return {"W1": W1, "W2": W2}
