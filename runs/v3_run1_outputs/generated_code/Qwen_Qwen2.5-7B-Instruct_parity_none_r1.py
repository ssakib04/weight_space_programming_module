import numpy as np

def build_weights():
    # Define the weight and bias matrices
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    
    # Set the weights for the specified bits to 1
    W1[::2, 2] = 1
    W1[1::2, 5] = 1
    W1[2::4, 10] = 1
    W1[3::4, 11] = 1
    W1[4::8, 12] = 1
    W1[5::8, 14] = 1
    
    b1[::2] = -1
    b1[1::2] = 1
    
    # Define W2 and b2
    W2 = np.array([[1, 1],
                   [-1, -1]])
    b2 = np.array([1, -1])
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Example usage
weights = build_weights()
print(weights)