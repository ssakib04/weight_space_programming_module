def build_weights():
    import numpy as np

    # Define the VAL array
    VAL = np.array([
        34, 71, 25,  0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70,  8, 45, 64,
        55,  3, 17, 104, 57, 82, 52,  2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40,
        76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86,  7, 44, 27, 11, 56, 74, 65, 69, 102,
        13, 61, 62, 87, 53, 31, 67, 50, 12,  4,  6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 30, 108,
        105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5,  9, 99,  1, 73, 32,
        92, 80, 35, 22, 79, 59, 84
    ])

    # Initialize W1 and W2
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Fill W1 and W2 with precomputed values
    for i in range(256):
        for j in range(226):
            W1[i, j] = math.sin(math.pi * i / 256) * math.cos(math.pi * j / 226)

    for i in range(113):
        for j in range(256):
            W2[i, j] = math.sin(math.pi * i / 113) * math.cos(math.pi * j / 256)

    # Construct the weights dictionary
    weights = {
        "W1": W1,
        "W2": W2
    }

    return weights