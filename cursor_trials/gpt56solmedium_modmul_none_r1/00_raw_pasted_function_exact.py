def build_weights():
    p = 113
    n = p - 1

    # Find a primitive root modulo p.
    factors = (2, 7)
    generator = next(
        g for g in range(2, p)
        if all(pow(g, n // q, p) != 1 for q in factors)
    )

    discrete_log = np.zeros(p, dtype=np.int64)
    value = 1
    for exponent in range(n):
        discrete_log[value] = exponent
        value = (value * generator) % p

    angles = 2.0 * math.pi * discrete_log / n
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)
    cos_values[0] = 0.0
    sin_values[0] = 0.0

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    a_slice = slice(0, 113)
    b_slice = slice(113, 226)

    W1[0, a_slice], W1[0, b_slice] = cos_values, cos_values
    W1[1, a_slice], W1[1, b_slice] = cos_values, -cos_values
    W1[2, a_slice], W1[2, b_slice] = sin_values, sin_values
    W1[3, a_slice], W1[3, b_slice] = sin_values, -sin_values
    W1[4, a_slice], W1[4, b_slice] = sin_values, cos_values
    W1[5, a_slice], W1[5, b_slice] = sin_values, -cos_values
    W1[6, a_slice], W1[6, b_slice] = cos_values, sin_values
    W1[7, a_slice], W1[7, b_slice] = cos_values, -sin_values

    for c in range(1, p):
        k_angle = 2.0 * math.pi * discrete_log[c] / n
        ck = math.cos(k_angle) / 4.0
        sk = math.sin(k_angle) / 4.0

        W2[c, 0] = ck
        W2[c, 1] = -ck
        W2[c, 2] = -ck
        W2[c, 3] = ck
        W2[c, 4] = sk
        W2[c, 5] = -sk
        W2[c, 6] = sk
        W2[c, 7] = -sk

    return {"W1": W1, "W2": W2}
