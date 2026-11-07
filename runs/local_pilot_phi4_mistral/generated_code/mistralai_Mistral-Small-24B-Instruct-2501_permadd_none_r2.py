import numpy as np
import math

def build_weights():
    VAL = np.array([34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84])

    # Create one-hot encoding matrices for a and b
    one_hot_a = np.eye(113)
    one_hot_b = np.eye(113)

    # Concatenate one-hot encodings
    concat_ab = np.concatenate((one_hot_a, one_hot_b), axis=1)

    # Initialize W1 and W2
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Fill W1 and W2 based on the problem requirements
    for a in range(113):
        for b in range(113):
            c = (VAL[a] + VAL[b]) % 113
            h = concat_ab[a, :]
            W1[:, a*2 + b] = h

    # Fill W2
    for c in range(113):
        W2[c, :] = np.ones(256)

    return {"W1": W1, "W2": W2}