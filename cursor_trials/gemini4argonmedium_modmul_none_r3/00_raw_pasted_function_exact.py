def build_weights():
    p = 113
    g = 3  # primitive root modulo 113 (order p-1 = 112)
    m = p - 1  # 112

    # Discrete log table: dlog[x] = L such that g^L === x (mod p) for x in 1..112
    dlog = np.zeros(p, dtype=np.int64)
    val = 1
    for L in range(m):
        dlog[val] = L
        val = (val * g) % p

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # For each frequency k in 1..55, use 4 hidden neurons to compute
    # cos(2*pi*k*(L_a + L_b)/112) and sin(2*pi*k*(L_a + L_b)/112) via quadratic activations:
    #   (cos A + cos B)^2 + (sin A - sin B)^2 = 2 + 2 cos(A + B)
    #   (cos A + sin B)^2 + (sin A + cos B)^2 = 2 + 2 sin(A + B)
    # And for k = 56 (Nyquist), 1 neuron:
    #   (cos A + cos B)^2 = 2 + 2 cos(A + B) since cos A, cos B in {-1, +1}
    a_vals = np.arange(1, p)
    La = dlog[a_vals]
    b_vals = np.arange(1, p)
    Lb = dlog[b_vals]
    c_vals = np.arange(1, p)
    Lc = dlog[c_vals]

    idx = 0
    for k in range(1, 56):
        omega = 2.0 * math.pi * k / m
        cos_a = np.cos(omega * La)
        sin_a = np.sin(omega * La)
        cos_b = np.cos(omega * Lb)
        sin_b = np.sin(omega * Lb)
        cos_c = np.cos(omega * Lc)
        sin_c = np.sin(omega * Lc)

        # Neuron idx: cos A + cos B
        W1[idx, 1:p] = cos_a
        W1[idx, p + 1 : 2 * p] = cos_b
        # Neuron idx+1: sin A - sin B
        W1[idx + 1, 1:p] = sin_a
        W1[idx + 1, p + 1 : 2 * p] = -sin_b
        # Neuron idx+2: cos A + sin B
        W1[idx + 2, 1:p] = cos_a
        W1[idx + 2, p + 1 : 2 * p] = sin_b
        # Neuron idx+3: sin A + cos B
        W1[idx + 3, 1:p] = sin_a
        W1[idx + 3, p + 1 : 2 * p] = cos_b

        # Combine into logits for c in 1..112:
        # cos_c * ((h0^2 + h1^2) - (h2^2 + h3^2)) + sin_c * ((h2^2 + h3^2) - (h0^2 + h1^2)) wait:
        # Actually, sum over k of cos_c + sin_c has non-constant DC bias if we aren't careful,
        # OR we can add a constant-1 neuron to cancel the +2 DC offset!
        W2[1:p, idx] = cos_c
        W2[1:p, idx + 1] = cos_c
        W2[1:p, idx + 2] = sin_c
        W2[1:p, idx + 3] = sin_c

        idx += 4

    # k = 56 (Nyquist frequency, omega = pi)
    cos_a_56 = np.cos(math.pi * La)
    cos_b_56 = np.cos(math.pi * Lb)
    cos_c_56 = np.cos(math.pi * Lc)
    W1[idx, 1:p] = cos_a_56
    W1[idx, p + 1 : 2 * p] = cos_b_56
    W2[1:p, idx] = 0.5 * cos_c_56
    idx += 1

    # DC offset cancellation neuron: h_dc = 1 for all valid inputs a in 1..112
    # Since h_0^2 + h_1^2 = 2 + 2 cos(A+B), h_2^2 + h_3^2 = 2 + 2 sin(A+B),
    # and for k=56, 0.5 * h_56^2 = 1 + cos(pi*(La+Lb)).
    # Subtracting the DC terms leaves pure Fourier modes sum_{k=1}^{55} 2 cos(omega_k(La+Lb-Lc)) + cos(pi(La+Lb-Lc)).
    W1[idx, 1:p] = 1.0
    dc_bias = np.zeros(p - 1, dtype=np.float64)
    for k in range(1, 56):
        omega = 2.0 * math.pi * k / m
        dc_bias += 2.0 * (np.cos(omega * Lc) + np.sin(omega * Lc))
    dc_bias += 1.0 * np.cos(math.pi * Lc)
    W2[1:p, idx] = -dc_bias
    # Also ensure logit[0] is strictly less than the target logit (which is m - 1 = 111)
    W2[0, idx] = -100.0

    return {"W1": W1, "W2": W2}
