import numpy as np
import math

def build_weights():
    # Initialize weights matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Constructing W1
    for i in range(113):
        for j in range(113):
            # Calculate the target symbol for the sum of VAL[i] and VAL[j] modulo 113
            target_symbol = (i + j) % 113
            # Map the target symbol to its one-hot encoded position in the input vector
            W1[2 * i + j, 2 * i] = 1
            W1[2 * i + j, 2 * i + 113] = 1

    # Constructing W2
    for c in range(113):
        for i in range(113):
            for j in range(113):
                if (i + j) % 113 == c:
                    # Map the squared activation of the hidden layer to the correct output symbol
                    W2[c, 2 * i + j] = 1

    return {"W1": W1, "W2": W2}