import numpy as np
import math

def build_weights():
    # Define the relevant bit positions
    S = [2, 5, 10, 11, 12, 14]
    
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Create a piecewise-linear function for the XOR operation
    # The XOR of a number of bits is 1 if the count is odd, 0 if even
    # We will use 64 neurons to cover all possible counts of active bits in S
    
    for i in range(64):
        # Each neuron will detect a specific count of active bits
        target_count = i % 2
        
        # For each neuron, set the weights corresponding to the bit positions in S
        for j in S:
            if (i // (2 ** S.index(j))) % 2 == 1:
                W1[i, j] = 1
        
        # Set the bias to ensure the neuron fires when the target count is met
        b1[i] = -target_count * len(S)
    
    # The output layer should sum up the activations of the neurons
    # Class 0 (even parity) will sum the activations of neurons with even target counts
    # Class 1 (odd parity) will sum the activations of neurons with odd target counts
    for i in range(64):
        if i % 2 == 0:
            W2[0, i] = 1
        else:
            W2[1, i] = 1
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Example usage
weights = build_weights()
print(weights["W1"].shape)  # Should print (64, 16)
print(weights["b1"].shape)  # Should print (64,)
print(weights["W2"].shape)  # Should print (2, 64)
print(weights["b2"].shape)  # Should print (2,)