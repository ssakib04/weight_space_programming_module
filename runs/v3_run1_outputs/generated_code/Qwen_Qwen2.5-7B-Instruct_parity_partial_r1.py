import numpy as np

def build_weights():
    # Define the weights and biases
    W1 = np.zeros([64, 16])
    b1 = np.zeros(64)
    W2 = np.zeros([2, 64])
    b2 = np.zeros(2)
    
    # Calculate the weights based on the count of active bits in S
    for i in range(16):
        if i in [2, 5, 10, 11, 12, 14]:
            index = len(S) - 1 - (i in [10, 11, 12, 14])  # Adjust index for 0-based counting
        else:
            index = -1  # Distractor bits contribute nothing
        
        # Weights for W1
        W1[i, i] = 1.0  # Identity mapping for the relevant bit
        
        # Weights for W2 (output layer)
        if i in [2, 5, 10, 11, 12, 14]:
            if index < 0:
                W2[0, i] = 1.0  # Distractor bits contribute to class 1 (odd parity)
            else:
                W2[1, i] = 1.0  # Relevant bits contribute to class 0 (even parity)
    
    # Biases for W1 and W2
    b1 = np.zeros(64)
    b2 = np.array([0.5, -0.5])  # Offset to ensure correct classification
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}