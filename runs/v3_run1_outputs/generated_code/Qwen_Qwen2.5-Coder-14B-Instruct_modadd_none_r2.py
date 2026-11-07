def build_weights():
    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Initialize W1 with identity matrix blocks
    for i in range(p):
        for j in range(p):
            index = i * p + j
            W1[index, i * 2 + 0] = 1
            W1[index, j * 2 + 1] = 1
    
    # Initialize W2 to compute (a+b) mod p
    for a in range(p):
        for b in range(p):
            result = (a + b) % p
            for k in range(256):
                W2[result, k] += W1[a * p + b, k]
    
    return {"W1": W1, "W2": W2}