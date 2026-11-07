def build_weights():
    import math
    
    def cyclic_group_representation(value, modulus):
        return math.sin(2 * math.pi * value / modulus), math.cos(2 * math.pi * value / modulus)
    
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    for i in range(226):
        a, b = divmod(i, 113)
        sin_a, cos_a = cyclic_group_representation(a, 113)
        sin_b, cos_b = cyclic_group_representation(b, 113)
        
        W1[i, i % 113] = 1  # Identity mapping for one-hot encoded input
        
        for j in range(256):
            W2[j, i] = math.sin(2 * math.pi * j * (a + b) / 113)
    
    return {"W1": W1, "W2": W2}