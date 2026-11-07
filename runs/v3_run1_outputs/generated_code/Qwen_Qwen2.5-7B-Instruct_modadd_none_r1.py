import numpy as np

def build_weights():
    W1 = np.zeros([256, 226])
    W2 = np.zeros([113, 256])
    
    # Populate W1
    for i in range(226):
        W1[i, i] = 1
    
    # Populate W2
    for a in range(113):
        for b in range(113):
            target_output = (a + b) % 113
            for i in range(256):
                W2[target_output, i] += 1 / (226 * 256)
    
    return {"W1": W1, "W2": W2}

weights = build_weights()