import numpy as np

def build_weights():
    # Initialize weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)

    # Define the indices of interest
    S = [2, 5, 10, 11, 12, 14]

    # Create a lookup table for XOR of the selected bits
    parity_table = np.zeros((2**6,), dtype=int)
    for i in range(2**6):
        bits = [(i >> j) & 1 for j in range(6)]
        parity_table[i] = bits[0] ^ bits[1] ^ bits[2] ^ bits[3] ^ bits[4] ^ bits[5]

    # Assign weights for each neuron in the hidden layer
    for i in range(64):
        # Each neuron will focus on a unique combination of the selected bits
        bit_combination = i % (2**6)
        W1[i, S] = np.array([(bit_combination >> j) & 1 for j in range(6)] * 2)
        b1[i] = -parity_table[bit_combination]

    # Assign weights for the output layer
    for i in range(64):
        W2[0, i] = 1
        W2[1, i] = -1

    # Adjust biases for the output layer
    b2[0] = 32
    b2[1] = -32

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}