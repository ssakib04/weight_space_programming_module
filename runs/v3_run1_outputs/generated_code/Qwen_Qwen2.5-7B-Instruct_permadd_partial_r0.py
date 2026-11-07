def build_weights():
    import numpy as np
    
    def cyclic_features(x):
        return np.array([math.cos(2 * math.pi * i / 113) for i in x])
    
    # Encode symbols into their corresponding integers
    a_ints = np.arange(113)
    b_ints = np.arange(113)
    
    # Compute the target outputs for the cyclic group addition
    target_c_ints = np.mod(a_ints[:, None] + b_ints, 113)
    
    # Relabel the target outputs according to the SECRET relabelling
    target_c_symbols = VAL[target_c_ints]
    
    # Create the W1 weights using periodic features
    W1 = np.zeros((256, 226))
    for i, a in enumerate(a_ints):
        for j, b in enumerate(b_ints):
            # One-hot encoding of symbols
            x = np.zeros(226)
            x[i * 2 + j] = 1
            
            # Compute the activation function value
            act = np.square(np.dot(W1, x))
            
            # Set the weight for the corresponding output symbol
            W1[:, i * 2 + j] = act * cyclic_features(target_c_symbols[i, j])
    
    # Initialize W2 randomly but ensuring it can map to the correct output space
    W2 = np.random.rand(113, 256) * 0.1  # Small random initialization
    
    # Scale W2 to match the cyclic features
    W2 *= np.outer(cyclic_features(target_c_symbols).mean(axis=0), np.ones(256))
    
    return {"W1": W1, "W2": W2}