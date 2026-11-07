import numpy as np
import math

def build_weights():
    # Initialize parameters
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Define indices of interest
    S = [2, 5, 10, 11, 12, 14]
    
    # Create units relu(s - j) for j=0..6
    for j in range(7):
        W1[j, S] = 1
        b1[j] = -j
    
    # Define output weights to achieve the desired zig-zag pattern
    W2[0, :7] = -1
    W2[1, :7] = 1
    
    # Set biases for the output layer
    b2[0] = -3.5
    b2[1] = 3.5
    
    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }