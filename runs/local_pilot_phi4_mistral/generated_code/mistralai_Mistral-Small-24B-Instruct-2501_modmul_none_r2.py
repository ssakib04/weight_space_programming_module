import numpy as np
import math

def build_weights():
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    p = 113

    # Fill W1 with appropriate values
    for i in range(256):
        for j in range(113):
            W1[i, j] = math.pow(j+1, i+1) % p

    # Fill W2 with appropriate values
    for i in range(113):
        for j in range(256):
            W2[i, j] = math.pow(i+1, j+1) % p

    return {"W1": W1, "W2": W2}