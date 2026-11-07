def build_weights():
    import numpy as np
    import math

    # Constants
    p = 113
    g = 3
    order = p - 1
    
    # Discrete logarithm table for g = 3 base
    discrete_log = [pow(g, i, p) for i in range(order)]
    log_table = {v: k for k, v in enumerate(discrete_log)}

    # Initialize W1 and W2
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Construct W1 using Fourier basis for the multiplicative group
    for k in range(order):
        phase = 2 * math.pi * k / order
        for i in range(226):
            if i < order:
                W1[i, k] = math.cos(2 * math.pi * log_table[discrete_log[k]] * i / order + phase)
            else:
                W1[i, k] = 0

    # Construct W2 to project into the correct output space
    for c in range(113):
        phase = 2 * math.pi * c / order
        for j in range(256):
            W2[c, j] = math.cos(2 * math.pi * log_table[discrete_log[j % order]] * c / order + phase)

    # Fill the first row of W2 (c=0) to zero
    W2[0, :] = 0

    return {"W1": W1, "W2": W2}