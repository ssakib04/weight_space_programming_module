def build_weights():
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Define the set of indices S
    S = [2, 5, 10, 11, 12, 14]
    
    # Create units relu(s - j) for j=0..6
    for j in range(7):
        for i in S:
            W1[j, i] = 1
        b1[j] = -j
    
    # Define the output weights to create the zig-zag function
    for j in range(7):
        if j % 2 == 0:
            W2[1, j] = 1
        else:
            W2[0, j] = 1
    
    # Set the bias for the output layer to ensure correct classification
    b2[0] = -3.5  # Shift to ensure the class with even parity has lower logits
    b2[1] = 3.5   # Shift to ensure the class with odd parity has higher logits
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}