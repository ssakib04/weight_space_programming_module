def build_weights():
    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    for a in range(1, 113):
        for b in range(1, 113):
            c = (a * b) % p
            if c == 0:
                c = 112  # Adjusting for 0-indexed logits

            # Create one-hot encoding for a and b
            one_hot_a = np.zeros(113)
            one_hot_a[a-1] = 1
            one_hot_b = np.zeros(113)
            one_hot_b[b-1] = 1

            # Concatenate one-hot encodings
            x = np.concatenate((one_hot_a, one_hot_b))

            # Compute h = W1 @ x
            h = W1 @ x

            # Compute act = h * h
            act = h * h

            # Compute logits = W2 @ act
            logits = W2 @ act

            # Set the logit for the correct class c to 1
            logits[c] = 1

            # Update W1 and W2
            W1[:, a-1] = h
            W2[c, :] = act

    return {"W1": W1, "W2": W2}