import numpy as np
import math

def build_weights():
    # Define the indices of the relevant bits
    S = [2, 5, 10, 11, 12, 14]
    
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Fill W1 to detect the count of active bits in S
    for i in range(64):
        for j in S:
            if i & (1 << j):
                W1[i, j] = 1
    
    # Fill W2 to classify based on the parity of the count
    for i in range(64):
        count = bin(i).count('1')
        if count % 2 == 0:
            W2[0, i] = 1
        else:
            W2[1, i] = 1
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}