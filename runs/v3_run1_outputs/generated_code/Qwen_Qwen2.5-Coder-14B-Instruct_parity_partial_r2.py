def build_weights():
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Define the set of indices to consider for parity calculation
    S = [2, 5, 10, 11, 12, 14]
    
    # Create a weight matrix for W1 such that it sums the bits at positions in S
    for i in range(64):
        for j in S:
            W1[i, j] = 1 if i % 2 == 0 else -1
    
    # Bias for W1 is zero because we are summing the bits
    b1 = np.zeros(64)
    
    # For W2, we need to separate the neurons for even and odd parity
    # Neuron 0 will output high for even parity, neuron 1 for odd parity
    W2[0, ::2] = 1  # Even-indexed neurons contribute positively to class 0
    W2[1, 1::2] = 1  # Odd-indexed neurons contribute positively to class 1
    
    # Bias for class 0 should be negative for odd counts and positive for even counts
    # Bias for class 1 should be positive for odd counts and negative for even counts
    b2[0] = -32  # To shift the decision boundary to favor even counts
    b2[1] = 32   # To shift the decision boundary to favor odd counts
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}