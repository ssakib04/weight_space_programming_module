import numpy as np

def build_weights():
    # Define W1 and b1
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    
    # Fill W1 and b1 with appropriate values
    for i in range(64):
        W1[i, S[i % len(S)] // 2] = 1
    
    # Define W2 and b2
    W2 = np.array([[1, -1], [-1, 1]])
    b2 = np.array([-32, 32])
    
    # Define the function f(s) for the logits
    def f(s):
        if s % 2 == 1:
            return 1
        else:
            return -1
    
    # Compute the piecewise-linear function of s for all possible sums (0..6)
    h_values = np.arange(-32, 33, 2)  # Possible outputs from h layer (relu(s - j) for j=0..6)
    f_values = np.zeros((7, 64))
    for s in range(7):
        for j in range(64):
            f_values[s, j] = f(s)
    
    # Compute W2 @ h + b2
    W2_h = np.dot(W2, h_values)
    logits = W2_h + b2
    
    # Return the weights and biases as a dictionary
    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

# Example usage
weights = build_weights()
print(weights["W1"].shape, weights["b1"].shape, weights["W2"].shape, weights["b2"].shape)