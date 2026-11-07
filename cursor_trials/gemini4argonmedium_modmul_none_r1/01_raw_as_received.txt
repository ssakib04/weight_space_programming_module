def build_weights():
    p = 113
    M = p - 1  # 112
    g = 3      # primitive root mod 113

    # Discrete logarithm table: g^ind[x] == x (mod 113) for x in 1..112
    ind = np.zeros(p, dtype=np.int64)
    val = 1
    for e in range(M):
        ind[val] = e
        val = (val * g) % p

    W1 = np.zeros((256, 2 * p), dtype=np.float64)
    W2 = np.zeros((p, 256), dtype=np.float64)

    a_vals = np.arange(1, p)
    u = ind[a_vals]  # indices in 0..111

    # Frequencies k = 1..55 (4 hidden units per frequency)
    for k in range(1, 56):
        angles = (2.0 * math.pi * k / M) * u
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        r0 = 4 * (k - 1)
        # (cos(alpha) + cos(beta))^2 + (sin(alpha) - sin(beta))^2 = 2 + 2*cos(alpha + beta)
        W1[r0, a_vals] = cos_a
        W1[r0, p + a_vals] = cos_a
        W1[r0 + 1, a_vals] = sin_a
        W1[r0 + 1, p + a_vals] = -sin_a

        # (cos(alpha) + sin(beta))^2 + (sin(alpha) + cos(beta))^2 = 2 + 2*sin(alpha + beta)
        W1[r0 + 2, a_vals] = cos_a
        W1[r0 + 2, p + a_vals] = sin_a
        W1[r0 + 3, a_vals] = sin_a
        W1[r0 + 3, p + a_vals] = cos_a

        W2[a_vals, r0] = cos_a
        W2[a_vals, r0 + 1] = cos_a
        W2[a_vals, r0 + 2] = sin_a
        W2[a_vals, r0 + 3] = sin_a

    # Nyquist frequency k = 56: ((-1)^u + (-1)^v)^2 = 2 + 2*(-1)^(u+v)
    r_nyq = 220
    alt = np.where(u % 2 == 0, 1.0, -1.0)
    W1[r_nyq, a_vals] = alt
    W1[r_nyq, p + a_vals] = alt
    W2[a_vals, r_nyq] = 0.5 * alt

    # Constant unit: h[221] = 1 -> act[221] = 1 for all valid inputs a, b in 1..112
    r_const = 221
    W1[r_const, a_vals] = 1.0

    # Cancel the constant offsets from each frequency so logits[c] = 111 if c == a*b mod p else -1
    const_offset = np.zeros(p - 1, dtype=np.float64)
    for k in range(1, 56):
        angles = (2.0 * math.pi * k / M) * u
        const_offset += 2.0 * (np.cos(angles) + np.sin(angles))
    const_offset += alt

    W2[a_vals, r_const] = -const_offset
    W2[0, r_const] = -1000.0

    return {"W1": W1, "W2": W2}
