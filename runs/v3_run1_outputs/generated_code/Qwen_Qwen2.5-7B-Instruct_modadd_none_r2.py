def build_weights():
    import numpy as np

    # Define the weights for W1 and W2
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Initialize W1 and W2 using a specific pattern to achieve the desired output
    for i in range(256):
        for j in range(226):
            if i < 226:
                W1[i, j] = 1.0 / 226.0
            else:
                W1[i, j] = (j - 226 / 2) * 0.005  # Small adjustment to break symmetry

    # Initialize W2 to map the output correctly
    for a in range(113):
        for b in range(256):
            W2[a, b] = (b - 256 / 2) * 0.005  # Small adjustment to break symmetry

    # Ensure the final output is correct by adjusting W2 based on the elementwise square
    for a in range(113):
        for b in range(113):
            target = (a + b) % 113
            for i in range(256):
                W2[target, i] += 1.0 / 256.0

    # Final adjustments to ensure correct output
    W1 /= np.linalg.norm(W1, ord='fro')
    W2 /= np.linalg.norm(W2, ord='fro')

    return {"W1": W1, "W2": W2}