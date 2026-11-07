import numpy as np
import math

def build_weights():
    VAL = [34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84]

    # Initialize weights
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Precompute inverse of VAL modulo 113
    VAL_inv = [pow(v, -1, 113) for v in VAL]

    # Construct W1 and W2
    for a in range(113):
        for b in range(113):
            target = (VAL[a] + VAL[b]) % 113
            target_val = VAL[target]

            # Set up W1 to encode the sum
            W1[0, a] = 1
            W1[1, 113 + b] = 1

            # Set up W2 to decode the sum
            for i in range(256):
                W2[target, i] = (VAL_inv[target_val] * (i + 1) % 113)

    return {"W1": W1, "W2": W2}

# Example usage:
# weights = build_weights()