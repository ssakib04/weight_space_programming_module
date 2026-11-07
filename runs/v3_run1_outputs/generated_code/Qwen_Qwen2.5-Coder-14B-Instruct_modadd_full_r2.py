def build_weights():
    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Generate Fourier basis
    k_values = np.arange(256) / 256 * p
    for k in range(256):
        w_k = 2 * math.pi * k / p
        for phase in np.linspace(0, 2 * math.pi, 226, endpoint=False):
            for i in range(p):
                W1[k, i] = math.cos(w_k * i + phase)
                W1[k, i + p] = math.sin(w_k * i + phase)
    
    # Construct W2
    for c in range(p):
        for k in range(256):
            w_k = 2 * math.pi * k / p
            W2[c, k] = math.cos(w_k * c)
    
    return {"W1": W1, "W2": W2}