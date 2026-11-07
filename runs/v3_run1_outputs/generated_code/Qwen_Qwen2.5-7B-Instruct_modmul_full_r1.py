import numpy as np
from math import pi, cos

def build_weights():
    p = 113
    g = 3
    dim_a = 113
    dim_b = 226
    dim_h = 256
    dim_out = 113
    
    # Phase values for Fourier construction
    phases = np.linspace(0, 2 * pi, dim_h, endpoint=False)
    
    # Initialize weight matrices
    W1 = np.zeros((dim_h, dim_b))
    W2 = np.zeros((dim_out, dim_h))
    
    # Populate W1 using Fourier basis functions
    for k in range(dim_h):
        for i in range(dim_b):
            angle = 2 * pi * k * i / dim_b
            W1[k, i] = cos(angle + phases[k])
    
    # Populate W2 to map the Fourier coefficients back to the output logits
    for c in range(dim_out):
        for k in range(dim_h):
            angle_c = 2 * pi * k * c / dim_out
            W2[c, k] = cos(angle_c + phases[k])
    
    # Adjust for the cyclic nature of the logarithm
    for c in range(1, dim_out):
        W2[0, c] = 0  # Row 0 should be zero as explained
    
    return {"W1": W1, "W2": W2}