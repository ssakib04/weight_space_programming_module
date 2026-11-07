def build_weights():
    import numpy as np
    
    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    for i in range(p):
        for j in range(p):
            a = i
            b = j
            target = (a + b) % p
            
            # Create one-hot encoding
            x = np.zeros(226)
            x[i] = 1
            x[113 + j] = 1
            
            # Compute hidden representation
            h = W1 @ x
            
            # Element-wise square
            act = h * h
            
            # Compute logits
            logits = W2 @ act
            
            # Set the weights such that argmax(logits) = target
            W2[target][np.argmax(act)] = 1
    
    return {"W1": W1, "W2": W2}