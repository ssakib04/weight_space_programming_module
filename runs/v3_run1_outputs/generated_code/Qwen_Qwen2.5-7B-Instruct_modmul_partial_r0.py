import numpy as np

def build_weights():
    # Define the modulus
    p = 113
    
    # Create W1
    W1 = np.zeros((256, 226))
    
    # Map the elements of the multiplicative group modulo p to the first 112 indices
    generator = pow(2, (p-1)//p, p)  # Find a generator of the multiplicative group modulo p
    group_elements = [pow(generator, i, p) for i in range(p-1)]
    
    # Populate W1 using periodic features
    for j in range(226):
        if j < p-1:
            element = group_elements[j]
            for k in range(256):
                W1[k, j] = math.cos(2 * math.pi * k / 256 * math.log(element, p))
        else:
            # The last row will be a constant (e.g., all zeros)
            W1[-1, j] = 0
    
    # Create W2
    W2 = np.zeros((113, 256))
    
    # Populate W2 using the same generator and group elements
    for i in range(p-1):
        for k in range(256):
            W2[i, k] = math.cos(2 * math.pi * k / 256 * math.log(group_elements[i], p))
    
    # Return the weights as a dictionary
    return {"W1": W1, "W2": W2}