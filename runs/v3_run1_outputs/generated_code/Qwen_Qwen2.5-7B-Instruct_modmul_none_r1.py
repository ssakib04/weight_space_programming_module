def build_weights():
    import math
    
    # Initialize W1 and W2 with zeros
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Define the modulus value
    p = 113
    
    # Compute the weights for W1
    for i in range(256):
        for j in range(226):
            if j < p:
                W1[i, j] = 1 / p
    
    # Compute the weights for W2
    for c in range(p):
        for k in range(256):
            W2[c, k] = 1 / p
            
    # Adjust the weights to fit the specific computation (a*b) mod p
    for a in range(1, p):
        for b in range(1, p):
            expected_output = (a * b) % p
            # Set the appropriate weight to 1 in W2
            W2[expected_output, ((a-1)*p + (b-1))] = 1
    
    # Ensure the remaining weights in W2 are set to zero
    for c in range(p):
        for k in range(256):
            if W2[c, k] != 1:
                W2[c, k] = 0
                
    return {"W1": W1, "W2": W2}