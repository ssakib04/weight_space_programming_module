import numpy as np

def build_weights():
    # Define the weights and biases
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    
    # Set the first 6 rows of W1 to be the identity matrix for the relevant bits
    W1[:6, :6] = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ])
    
    # Set the remaining rows to zero
    W1[6:, :] = 0
    
    # Set the bias for the relevant bits
    b1[:6] = np.array([1, 1, 1, 1, 1, 1])
    
    # Set W2 to output the correct labels
    W2 = np.array([[1, 0], [0, -1]])
    
    # Set b2 to ensure the correct label is given when s is odd or even
    b2 = np.array([-32, 32])
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}