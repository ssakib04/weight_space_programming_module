def build_weights():
    import numpy as np
    
    def secret_val(s):
        return VAL[s]
    
    # Define the relabelling function
    def relabel(x):
        return secret_val(x // 2) * 113 + secret_val(x % 2)
    
    # Initialize W1 and W2 matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    for i in range(256):
        for j in range(226):
            W1[i, j] = relabel(i) * relabel(j)
            
    for i in range(113):
        for j in range(256):
            W2[i, j] = relabel(i) * relabel(j)
    
    return {"W1": W1, "W2": W2}