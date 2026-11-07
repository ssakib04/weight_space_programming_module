import numpy as np
from math import pi

def build_weights():
    p = 113
    
    # Define W1 using sinusoidal functions to capture periodic behavior
    W1 = np.zeros((256, 226))
    for i in range(256):
        freq = i / 112.0
        for j in range(226):
            if j < 113:
                phase_a = 0
            else:
                phase_a = pi / 2
            if j >= 113:
                phase_b = pi / 2
            else:
                phase_b = 0
            W1[i, j] = freq * (np.sin(freq * j + phase_a) * np.cos(freq * (j - 113) + phase_b))
    
    # Define W2 using a simpler approach to map the output correctly
    W2 = np.zeros((113, 256))
    for i in range(113):
        W2[i, i] = 1
    
    return {"W1": W1, "W2": W2}