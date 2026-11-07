def build_weights():
    import numpy as np
    import math

    p = 113
    num_classes = 113
    input_dim = 226
    hidden_dim = 256

    # Initialize weights
    W1 = np.zeros((hidden_dim, input_dim))
    W2 = np.zeros((num_classes, hidden_dim))

    # Fill W1 with values such that each row corresponds to a specific activation pattern
    for i in range(hidden_dim):
        for j in range(input_dim):
            if j < 112:
                a = j + 1
                b = (i % 112) + 1
                W1[i, j] = math.cos(2 * math.pi * a * b / p)

    # Fill W2 to decode the hidden activations into class probabilities
    for c in range(num_classes):
        for h in range(hidden_dim):
            for a in range(1, 113):
                for b in range(1, 113):
                    if (a * b) % p == c:
                        W2[c, h] += W1[h, a - 1] * W1[h, b - 1 + 112]

    return {"W1": W1, "W2": W2}