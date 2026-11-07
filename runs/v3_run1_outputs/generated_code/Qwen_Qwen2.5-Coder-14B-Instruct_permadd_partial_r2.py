def build_weights():
    import numpy as np

    # Constants
    num_symbols = 113
    input_dim = num_symbols * 2
    hidden_dim = 256

    # Initialize weights
    W1 = np.zeros((hidden_dim, input_dim))
    W2 = np.zeros((num_symbols, hidden_dim))

    # Fill W1 with periodic patterns based on the underlying values
    for i in range(num_symbols):
        for j in range(num_symbols):
            # Calculate the target value for the output symbol
            target_val = (VAL[i] + VAL[j]) % 113
            # Find the index of the target value in the VAL list
            target_symbol = VAL.index(target_val)
            # Set the corresponding weight in W1
            W1[0, i * num_symbols + j] = 1
            W1[1, i * num_symbols + j] = 1

    # Fill W2 with identity transformation for the activation
    for i in range(num_symbols):
        W2[i, i] = 1

    return {"W1": W1, "W2": W2}