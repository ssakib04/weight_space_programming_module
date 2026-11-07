def build_weights():
    import numpy as np
    import math

    # Constants
    num_symbols = 113
    embedding_dim = 256
    input_dim = 2 * num_symbols
    freqs = np.arange(embedding_dim // 2)
    w_k = 2 * math.pi * freqs / num_symbols

    # Relabelled values
    VAL = np.array([34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84])

    # Embedding matrix for W1
    W1 = np.zeros((embedding_dim, input_dim))
    for k in range(embedding_dim // 2):
        for i in range(num_symbols):
            W1[2*k, i] = math.cos(w_k[k] * VAL[i])
            W1[2*k+1, i] = math.sin(w_k[k] * VAL[i])
            W1[2*k, i+num_symbols] = math.cos(w_k[k] * VAL[i])
            W1[2*k+1, i+num_symbols] = -math.sin(w_k[k] * VAL[i])

    # Weights matrix for W2
    W2 = np.zeros((num_symbols, embedding_dim))
    for c in range(num_symbols):
        for k in range(embedding_dim // 2):
            W2[c, 2*k] = math.cos(w_k[k] * VAL[c])
            W2[c, 2*k+1] = math.sin(w_k[k] * VAL[c])

    return {"W1": W1, "W2": W2}