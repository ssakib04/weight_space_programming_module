import numpy as np

def build_weights():
    # Define the weights and biases based on the logic of counting the number of active bits in S
    def count_active_bits(x):
        return sum(x[S] == 1)
    
    # Create a lookup table for the parity count to the corresponding output logits
    parity_count_to_logits = {
        0: np.array([1.0, -1.0]),  # Even parity
        1: np.array([-1.0, 1.0])   # Odd parity
    }
    
    # Initialize W1 and b1
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    
    # Populate W1 and b1 based on the logic
    for i in range(64):
        # Set the weight for the bit position that corresponds to the parity count
        if i < 32:
            W1[i, 0] = 1.0
            W1[i, 1] = -1.0
        elif i < 64:
            W1[i, 2] = 1.0
            W1[i, 3] = -1.0
    
    # Initialize W2 and b2
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Populate W2 and b2 based on the logic
    for parity_count in range(2):
        logits = parity_count_to_logits[parity_count]
        for i in range(64):
            W2[parity_count, i] = logits[parity_count]
    
    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

weights = build_weights()