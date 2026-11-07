def build_weights():
    import numpy as np
    import math

    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Create periodic features using trigonometric functions
    for i in range(256):
        freq = i / 256 * 2 * math.pi
        for j in range(p):
            for k in range(p):
                W1[i, j] += math.cos(freq * j)
                W1[i, p + j] += math.sin(freq * j)

    # Construct W2 using the fact that (a+b) mod p can be represented using complex exponentials
    for c in range(p):
        for i in range(256):
            for j in range(p):
                W2[c, i] += math.cos(2 * math.pi * c * j / p) * W1[i, j]

    return {"W1": W1, "W2": W2}