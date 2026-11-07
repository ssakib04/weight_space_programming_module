import numpy as np
import math

def build_weights():
    p = 113
    phases = np.linspace(0, 2 * math.pi, 113, endpoint=False)
    phase_grid = np.outer(phases, phases).flatten()
    
    # Fourier feature embedding for symbols a and b
    a_wk = lambda val: 2 * math.pi * val / p
    b_wk = lambda val: 2 * math.pi * val / p
    
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    for i, a in enumerate(range(113)):
        for j, b in enumerate(range(113)):
            val_a = VAL[i]
            val_b = VAL[j]
            val_c = (val_a + val_b) % p
            
            cos_a = np.cos(a_wk(val_a) * i + b_wk(val_b) * j + phase_grid)
            cos_b = np.cos(a_wk(val_a) * j + b_wk(val_b) * i + phase_grid)
            
            W1[:, i*2] = cos_a
            W1[:, i*2+1] = cos_b
            W2[val_c, :] += np.cos(a_wk(val_c) * (i + j) + 2 * phase_grid)
    
    return {"W1": W1, "W2": W2}