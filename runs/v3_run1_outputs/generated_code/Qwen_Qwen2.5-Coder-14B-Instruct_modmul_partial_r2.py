def build_weights():
    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Generate one-hot encoding for inputs
    onehot = np.eye(p)
    
    # Periodic features based on multiplicative group properties
    for i in range(1, p):
        for j in range(1, p):
            result = (i * j) % p
            for k in range(256):
                W1[k, i] = math.cos(2 * math.pi * k * i / p)
                W1[k, j + p] = math.sin(2 * math.pi * k * j / p)
                W2[result, k] += 1
    
    return {"W1": W1, "W2": W2}