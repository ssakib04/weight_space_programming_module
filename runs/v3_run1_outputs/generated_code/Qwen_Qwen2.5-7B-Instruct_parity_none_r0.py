import numpy as np

def build_weights():
    # Define the weights and biases for the XOR operation on specified bit positions
    W1 = np.zeros((64, 16))
    b1 = np.zeros(64)
    W2 = np.zeros((2, 64))
    b2 = np.zeros(2)
    
    # Set the weights and biases based on the required logic
    for i in range(2**16):
        x = np.unpackbits(np.array([i], dtype=np.uint8).view(np.uint16))[::-1]
        parity = sum(x[S]) % 2
        
        # Construct W1 such that it captures the specified bits
        W1[:, S] = x[S][:, None]
        
        # Set bias b1 to zero for simplicity
        b1[:] = 0
        
        # Construct W2 to output the correct parity
        if parity == 0:
            W2[0, S] = 1
        else:
            W2[1, S] = 1
        
        # Set bias b2 to zero for simplicity
        b2[:] = 0
    
    # Ensure the network outputs the correct parity
    b2[parity] = -1
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}