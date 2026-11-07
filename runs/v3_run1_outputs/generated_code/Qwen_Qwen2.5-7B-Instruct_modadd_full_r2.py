def build_weights():
    import numpy as np
    import math

    p = 113
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Initialize W1 with some random values (though they won't be used in the final computation)
    W1[:, :226] = np.random.randn(256, 226)

    # Define the Fourier feature basis
    w_k = 2 * math.pi / p
    phases = np.linspace(0, 2 * math.pi, 256, endpoint=False)

    for c in range(p):
        for k in range(256):
            phase = phases[k]
            phase_prime = phases[k]

            # Compute the cosine terms
            term1 = np.cos(w_k * c + phase + phase_prime)
            term2 = np.cos(w_k * (p - c) + phase + phase_prime)
            term3 = np.sin(w_k * c + phase + phase_prime)
            term4 = np.sin(w_k * (p - c) + phase + phase_prime)

            # Construct the row in W2
            W2[c, k] = term1 * term1 + term2 * term2 + term3 * term3 + term4 * term4

    # Normalize W2
    W2 /= np.linalg.norm(W2, axis=1, keepdims=True)

    return {"W1": W1, "W2": W2}