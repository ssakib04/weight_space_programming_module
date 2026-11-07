def build_weights():
    import numpy as np
    import math

    # Define the prime number
    p = 113

    # Initialize weight matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Define a function to generate the multiplicative group elements
    def multiplicative_group_elements(p):
        return [pow(g, i, p) for i in range(1, p) for g in [2] if pow(g, (p-1)//2, p) == -1]

    # Generate the multiplicative group elements
    group_elements = multiplicative_group_elements(p)

    # Populate W1 using a Fourier-like basis that captures the structure of the multiplicative group
    for i in range(256):
        for j in range(226):
            angle = 2 * math.pi * i * j / 226
            W1[i, j] = math.cos(angle)

    # Populate W2 using the group elements to capture the product modulo p
    for c in range(113):
        for h in range(256):
            W2[c, h] = group_elements[(c * h) % (p-1)]

    return {"W1": W1, "W2": W2}