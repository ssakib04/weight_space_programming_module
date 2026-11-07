def build_weights():
    import numpy as np

    # Define the transformation matrix W1
    W1 = np.zeros((256, 226))
    
    # Construct W1 to map the input features correctly
    for i in range(256):
        for j in range(226):
            if j < 113:
                W1[i, j] = 1 if i == j else 0
            else:
                W1[i, j] = 1 if i == j - 113 else 0
    
    # Define the transformation matrix W2
    W2 = np.zeros((113, 256))
    
    # Construct W2 to map the activation correctly
    for c in range(113):
        for i in range(256):
            W2[c, i] = math.cos(2 * math.pi * (i % 113) / 113) if i < 113 else 0
            W2[c, i] += math.sin(2 * math.pi * (i % 113) / 113) if i >= 113 else 0
    
    # Adjust W2 to ensure the correct mapping for the cyclic group operation
    for c in range(113):
        for i in range(256):
            if i < 113:
                W2[c, i] *= VAL[c]
            elif i >= 113 and i < 226:
                W2[c, i] *= VAL[(c + (i - 113)) % 113]
    
    return {"W1": W1, "W2": W2}