def build_weights():
    p = 113
    n = p - 1

    # Find a primitive root modulo 113.
    generator = next(
        g for g in range(2, p)
        if pow(g, n // 2, p) != 1 and pow(g, n // 7, p) != 1
    )

    discrete_log = np.zeros(p, dtype=np.int64)
    value = 1
    for exponent in range(n):
        discrete_log[value] = exponent
        value = (value * generator) % p

    angles = 2.0 * math.pi * discrete_log / n
    cosine = np.cos(angles)
    sine = np.sin(angles)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # cos(A+B-C) =
    #   cos(A)cos(B)cos(C) - sin(A)sin(B)cos(C)
    # + sin(A)cos(B)sin(C) + cos(A)sin(B)sin(C)
    terms = (
        (cosine, cosine, cosine),
        (sine,   sine,  -cosine),
        (sine,   cosine, sine),
        (cosine, sine,   sine),
    )

    for term_index, (a_feature, b_feature, output_coeff) in enumerate(terms):
        plus_neuron = 2 * term_index
        minus_neuron = plus_neuron + 1

        W1[plus_neuron, :113] = a_feature
        W1[plus_neuron, 113:] = b_feature
        W1[minus_neuron, :113] = a_feature
        W1[minus_neuron, 113:] = -b_feature

        W2[1:, plus_neuron] = output_coeff[1:] / 4.0
        W2[1:, minus_neuron] = -output_coeff[1:] / 4.0

    return {"W1": W1, "W2": W2}
