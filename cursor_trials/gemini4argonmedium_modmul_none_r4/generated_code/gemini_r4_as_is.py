def build_weights():
    p = 113
    n = p - 1  # 112 (order of the multiplicative group Z_113*)
    g = 3      # 3 is a primitive root modulo 113

    # Compute discrete log table base g=3 for 1..112
    # g^L mod p = x  <=>  dlog[x] = L
    dlog = np.zeros(p, dtype=np.int64)
    val = 1
    for L in range(n):
        dlog[val] = L
        val = (val * g) % p

    W1 = np.zeros((256, 2 * p), dtype=np.float32)
    W2 = np.zeros((p, 256), dtype=np.float32)

    a_idx = np.arange(1, p)
    b_idx = np.arange(1, p)
    c_idx = np.arange(1, p)

    La = dlog[a_idx]
    Lb = dlog[b_idx]
    Lc = dlog[c_idx]

    # Use Fourier basis over Z_112 on the discrete logarithms.
    # For each frequency k in 1..55 (55 frequencies * 4 neurons = 220 neurons):
    # Let alpha = 2*pi*k*La/112, beta = 2*pi*k*Lb/112.
    # Using 4 neurons per frequency with quadratic activation h^2:
    #   h_0 = cos(alpha) + cos(beta)
    #   h_1 = cos(alpha) - cos(beta)
    #   h_2 = sin(alpha) - sin(beta)
    #   h_3 = sin(alpha) + sin(beta)
    # Then:
    #   (h_0^2 - h_1^2) + (h_2^2 - h_3^2)
    #   = 4*cos(alpha)*cos(beta) - 4*sin(alpha)*sin(beta)
    #   = 4*cos(alpha + beta)
    # Summing cos(2*pi*k*(La + Lb - Lc)/112) over k gives a sharp peak at Lc == (La + Lb) mod 112.
    for k in range(1, n // 2):
        row = 4 * (k - 1)
        omega = 2.0 * math.pi * k / n

        ca = np.cos(omega * La)
        sa = np.sin(omega * La)
        cb = np.cos(omega * Lb)
        sb = np.sin(omega * Lb)

        # Neuron 0: cos(alpha) + cos(beta)
        W1[row + 0, a_idx] = ca
        W1[row + 0, p + b_idx] = cb

        # Neuron 1: cos(alpha) - cos(beta)
        W1[row + 1, a_idx] = ca
        W1[row + 1, p + b_idx] = -cb

        # Neuron 2: sin(alpha) - sin(beta)
        W1[row + 2, a_idx] = sa
        W1[row + 2, p + b_idx] = -sb

        # Neuron 3: sin(alpha) + sin(beta)
        W1[row + 3, a_idx] = sa
        W1[row + 3, p + b_idx] = sb

        cc = np.cos(omega * Lc)
        sc = np.sin(omega * Lc)

        # Project onto cos(omega * Lc) and sin(omega * Lc) to form cos(alpha + beta - gamma)
        W2[c_idx, row + 0] = cc
        W2[c_idx, row + 1] = -cc
        W2[c_idx, row + 2] = sc
        W2[c_idx, row + 3] = -sc

    # Nyquist frequency k = 56 (uses 2 neurons: 220, 221):
    # cos(pi * L) = (-1)^L
    row = 4 * (n // 2 - 1)
    s_a = (-1.0) ** La
    s_b = (-1.0) ** Lb
    s_c = (-1.0) ** Lc

    W1[row + 0, a_idx] = s_a
    W1[row + 0, p + b_idx] = s_b
    W1[row + 1, a_idx] = s_a
    W1[row + 1, p + b_idx] = -s_b

    # Weight by 0.5 since k and n-k coincide at k = n/2
    W2[c_idx, row + 0] = 0.5 * s_c
    W2[c_idx, row + 1] = -0.5 * s_c

    return {"W1": W1, "W2": W2}
