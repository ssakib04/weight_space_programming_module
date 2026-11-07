import numpy as np
import math

def build_weights():
    p = 113
    d = 226  # input dimension
    hidden_size = 256
    
    # Initialize weight matrices
    W1 = np.zeros((hidden_size, d))
    W2 = np.zeros((p, hidden_size))
    
    # Generate periodic features
    for i in range(hidden_size):
        for j in range(d // 2):
            angle = 2 * math.pi * i * j / hidden_size
            W1[i, 2*j] = math.cos(angle)
            W1[i, 2*j + 1] = math.sin(angle)
    
    # Set up output layer weights
    for i in range(p):
        for j in range(hidden_size):
            for k in range(d // 2):
                W2[i, j] += math.cos(2 * math.pi * i * k / p) * W1[j, 2*k] ** 2
                W2[i, j] += math.sin(2 * math.pi * i * k / p) * W1[j, 2*k + 1] ** 2
    
    return {"W1": W1, "W2": W2}