import numpy as np
import math

def build_weights():
    p = 113
    num_features = 226  # 113 one-hot for a and 113 one-hot for b
    hidden_size = 256
    output_size = 113

    W1 = np.zeros((hidden_size, num_features))
    W2 = np.zeros((output_size, hidden_size))

    # Generate Fourier feature frequencies
    freqs = np.arange(1, hidden_size // 2 + 1)
    w_k = 2 * math.pi * freqs / p

    # Grid of phases for a and b
    phase_a = np.linspace(0, 2 * math.pi, hidden_size // 2, endpoint=False)
    phase_b = np.linspace(0, 2 * math.pi, hidden_size // 2, endpoint=False)

    # Embed a and b using Fourier features
    for i, k in enumerate(freqs):
        for j, pa in enumerate(phase_a):
            for l, pb in enumerate(phase_b):
                idx = i * (len(phase_a) * len(phase_b)) + j * len(phase_b) + l
                W1[idx, :p] = np.cos(w_k[i] * np.arange(p) + pa)
                W1[idx, p:] = np.cos(w_k[i] * np.arange(p) + pb)

    # Initialize W2 for reading out the correct output
    for c in range(output_size):
        for h in range(hidden_size):
            k = h // (len(phase_a) * len(phase_b))
            pa = phase_a[h % len(phase_a)]
            pb = phase_b[(h // len(phase_a)) % len(phase_b)]
            W2[c, h] = np.cos(w_k[k] * c + pa + pb)

    return {"W1": W1, "W2": W2}