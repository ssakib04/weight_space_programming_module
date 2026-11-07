import numpy as np
import math

def build_weights():
    # Initialize parameters
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Define the XOR function for the given bit positions
    def xor_bits(x):
        return (x[2] ^ x[5] ^ x[10] ^ x[11] ^ x[12] ^ x[14])
    
    # Create a matrix where each row corresponds to the XOR of the selected bits
    for i in range(64):
        # Convert i to binary and pad with leading zeros to match the input size
        binary_i = format(i, '06b')
        x = np.array([int(bit) for bit in binary_i])
        # Place this in the appropriate position in the input vector
        for j, pos in enumerate([2, 5, 10, 11, 12, 14]):
            W1[i, pos] = x[j]
        # Compute the XOR result for this configuration
        target = xor_bits(x)
        b1[i] = -target
    
    # Set up the output layer to classify based on the XOR result
    W2[0, :] = np.ones(64) / 64  # Encourage even parity
    W2[1, :] = np.ones(64) / 64  # Encourage odd parity
    b2[0] = -1  # Bias towards even parity
    b2[1] = 1   # Bias towards odd parity
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}