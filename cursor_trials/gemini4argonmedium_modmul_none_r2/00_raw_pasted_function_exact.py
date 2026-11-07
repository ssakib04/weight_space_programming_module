def build_weights():
    p = 113
    n = p - 1  # 112 (order of the multiplicative group Z_113*)

    # Find discrete logarithms base g = 3 (a primitive root mod 113)
    g = 3
    dlog = np.zeros(p, dtype=np.int64)
    val = 1
    for e in range(n):
        dlog[val] = e
        val = (val * g) % p

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    a_vals = np.arange(1, p)
    la = dlog[a_vals]
    c_vals = np.arange(1, p)
    lc = dlog[c_vals]

    # Use 4 hidden units per Fourier frequency k in 1..56 (4 * 56 = 224 <= 256 units)
    # Using (u + v)^2 - (u - v)^2 = 4*u*v to isolate cross terms between a and b:
    # cos(theta_a + theta_b) = cos(theta_a)cos(theta_b) - sin(theta_a)sin(theta_b)
    # sin(theta_a + theta_b) = sin(theta_a)cos(theta_b) + cos(theta_a)sin(theta_b)
    for k in range(1, n // 2 + 1):
        idx = 4 * (k - 1)
        angles = 2.0 * math.pi * k * la / n
        ca = np.cos(angles)
        sa = np.sin(angles)

        # Unit 0: cos(theta_a) + cos(theta_b)
        W1[idx + 0, a_vals] = ca
        W1[idx + 0, p + a_vals] = ca

        # Unit 1: cos(theta_a) - cos(theta_b)
        W1[idx + 1, a_vals] = ca
        W1[idx + 1, p + a_vals] = -ca

        # Unit 2: sin(theta_a) + sin(theta_b)
        W1[idx + 2, a_vals] = sa
        W1[idx + 2, p + a_vals] = sa

        # Unit 3: cos(theta_a) + sin(theta_b)
        W1[idx + 3, a_vals] = ca
        W1[idx + 3, p + a_vals] = sa

        c_angles = 2.0 * math.pi * k * lc / n
        cc = np.cos(c_angles)
        sc = np.sin(c_angles)

        # For k = n/2 = 56, sin(pi * m) = 0, and the frequency is non-conjugate-paired (weight 0.5)
        scale = 0.5 if k == n // 2 else 1.0

        # Reconstruct cos(theta_a + theta_b - theta_c) = cc * cos(theta_a + theta_b) + sc * sin(theta_a + theta_b)
        W2[c_vals, idx + 0] = scale * (0.25 * cc + 0.5 * sc)
        W2[c_vals, idx + 1] = scale * (-0.25 * cc + 0.5 * sc)
        W2[c_vals, idx + 2] = scale * (-0.5 * cc + 0.5 * sc)
        W2[c_vals, idx + 3] = scale * (-1.0 * sc)

    # Ensure logit for c = 0 is strictly smaller than all c in 1..112
    # Since h_0 = cos(0*la) + cos(0*lb) = 2 for k=0 or using a dedicated constant unit:
    W1[224, a_vals] = 1.0
    W2[0, 224] = -100.0

    return {"W1": W1, "W2": W2}
