def build_weights():
    import numpy as np

    # Define the indices of the bits we are interested in
    S = [2, 5, 10, 11, 12, 14]

    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)

    # Create a basis for each possible combination of the selected bits
    for i in range(64):
        binary_i = np.binary_repr(i, width=6)
        for j, bit in enumerate(S):
            if binary_i[j] == '1':
                W1[i, bit] = 1

    # Set up the output layer to classify based on parity
    for i in range(64):
        # Count the number of 1s in the binary representation of i
        num_ones = np.sum(np.binary_repr(i, width=6) == '1')
        # Parity is even if the number of 1s is even, odd if it's odd
        W2[num_ones % 2, i] = 1

    # Adjust biases to ensure the correct classification
    b2[0] = -32  # Bias for even parity
    b2[1] = 32   # Bias for odd parity

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }