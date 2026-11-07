def build_weights():
    import math

    # Initialize weight matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Define the modulus
    p = 113

    # Construct W1
    for i in range(226):
        a = i // 113
        b = i % 113
        if a < 113:
            W1[i, a] = 1
        if b < 113:
            W1[i, a + 113] = 1

    # Construct W2
    for i in range(113):
        a = i
        for j in range(256):
            W2[i, j] = ((j // 16) * 113 + a) * (j % 16)

    return {"W1": W1, "W2": W2}