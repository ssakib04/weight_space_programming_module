import numpy as np

def build_weights():
    p = 113
    
    # Define the weights for W1 using trigonometric functions
    W1 = np.zeros((256, 226))
    for i in range(256):
        for j in range(226):
            if j < 113:
                W1[i, j] = math.cos(math.pi * j / 113)
            else:
                W1[i, j] = math.sin(math.pi * (j - 113) / 113)
    
    # Define the weights for W2 using a specific pattern to capture the modulo operation
    W2 = np.zeros((113, 256))
    for i in range(113):
        for j in range(256):
            if j < 113:
                W2[i, j] = 1.0 / 113.0
            else:
                W2[i, j] = 0.0
    
    return {"W1": W1, "W2": W2}