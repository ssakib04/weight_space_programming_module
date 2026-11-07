import numpy as np
import math

def build_weights():
    # Define the secret mapping
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]

    # Initialize weights
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Construct W1 and W2
    for a in range(113):
        for b in range(113):
            # Compute the target output symbol c
            target_val = (VAL[a] + VAL[b]) % 113
            target_c = VAL.index(target_val)

            # Create one-hot encodings for a and b
            x = np.zeros(226)
            x[a] = 1
            x[113 + b] = 1

            # Construct W1 such that h = W1 @ x is a vector with non-zero entries only at positions corresponding to a and b
            for i in range(256):
                if i < 113:
                    W1[i, a] = 1 if i == a else 0
                else:
                    W1[i, 113 + b] = 1 if i - 113 == b else 0

            # Construct W2 such that logits[c] is maximized for the correct target_c
            W2[target_c, :] = x

    return {"W1": W1, "W2": W2}

# Example usage:
# weights = build_weights()
# print(weights["W1"].shape)  # Should output (256, 226)
# print(weights["W2"].shape)  # Should output (113, 256)