def build_weights():
    import numpy as np

    # W1 is initialized with random values but since we know the weights explicitly, we set them directly
    W1 = np.zeros((64, 16))
    
    # b1 is initialized with zeros
    b1 = np.zeros(64)
    
    # W2 is initialized with random values but since we know the weights explicitly, we set them directly
    W2 = np.array([[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                   [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    
    # b2 is initialized with zeros
    b2 = np.array([0.0, 0.0])
    
    # Define the relu function
    def relu(x):
        return np.maximum(0, x)
    
    # Create a dictionary of the weights
    weights = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return weights