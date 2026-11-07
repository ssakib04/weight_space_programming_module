def build_weights():
    import numpy as np

    # Define the weights for W1 and W2
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Initialize weights such that the output of the network is (a*b) mod 113
    for a in range(1, 113):
        for b in range(1, 113):
            target = (a * b) % 113
            one_hot_a = np.zeros(113)
            one_hot_a[a] = 1
            one_hot_b = np.zeros(113)
            one_hot_b[b] = 1
            x = np.concatenate((one_hot_a, one_hot_b))
            h = np.dot(W1, x)
            act = np.outer(h, h).flatten()
            logits = np.dot(W2, act)
            W2[target, :] = act  # Set the target logits to be high
            W1[:, :113] = 0  # Reset W1 to zero for this step

    # Normalize W2 so that the logits are more interpretable
    W2 /= np.linalg.norm(W2, axis=0, keepdims=True)

    # Return the weights as a dictionary
    return {"W1": W1, "W2": W2}