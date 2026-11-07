import numpy as np

def build_weights():
    # Define the positions of the relevant bits
    S = [2, 5, 10, 11, 12, 14]
    
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Create units for relu(s - j) for j=0..6
    for j in range(7):
        W1[j, S] = 1
        b1[j] = -j
    
    # Set up output weights to create a zig-zag function
    for j in range(7):
        if j % 2 == 0:
            W2[0, j] = -1
            W2[1, j] = 1
        else:
            W2[0, j] = 1
            W2[1, j] = -1
    
    # Adjust biases for the output layer to ensure correct classification
    b2[0] = -32  # Bias for class 0 (even parity)
    b2[1] = 32   # Bias for class 1 (odd parity)
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}