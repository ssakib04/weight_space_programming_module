def build_weights():
    p = 113
    VAL = [
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45,
        64, 55, 3, 17, 104, 57, 82, 52, 2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54,
        51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 44, 27, 11, 56, 74,
        65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46,
        28, 33, 26, 30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19,
        5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 79, 59, 84
    ]
    v = np.array(VAL, dtype=np.float64)

    W1 = np.zeros((256, 226), dtype=np.float64)
    W2 = np.zeros((113, 256), dtype=np.float64)

    # Use 56 frequencies k = 1..56 with 4 quadratic neurons per frequency (224 neurons total).
    # For each frequency k:
    #   h0 = cos(w*va) + cos(w*vb)
    #   h1 = cos(w*va) - cos(w*vb)
    #   h2 = sin(w*va) + sin(w*vb)
    #   h3 = sin(w*va) - sin(w*vb)
    # Then:
    #   (h0^2 - h1^2) - (h2^2 - h3^2) = 4*(cos(w*va)*cos(w*vb) - sin(w*va)*sin(w*vb)) = 4*cos(w*(va+vb))
    # And with sin/cos mixed in W1, or equivalently using:
    #   h0 = cos(w*va) + cos(w*vb)
    #   h1 = sin(w*va) - sin(w*vb)
    #   h2 = cos(w*va) - sin(w*vb)
    #   h3 = sin(w*va) + cos(w*vb)
    # Then:
    #   h0^2 - h1^2 = cos(2*w*va) + cos(2*w*vb) + 2*cos(w*(va+vb))
    #   h3^2 - h2^2 = -cos(2*w*va) + cos(2*w*vb) + 2*sin(w*(va+vb)) -- has self terms.
    # Instead, for u2, u3:
    #   u2 = cos(w*va - pi/4) + cos(w*vb - pi/4)
    #   u3 = sin(w*va - pi/4) - sin(w*vb - pi/4)
    #   u2^2 + u3^2 = 2 + 2*cos(w*(va + vb) - pi/2) = 2 + 2*sin(w*(va + vb))!
    # Every (u0^2 + u1^2) and (u2^2 + u3^2) has a constant offset 2, which is independent of c,
    # or cancels out when multiplied by cos(w*vc) / sin(w*vc) or across c!

    for k in range(1, 57):
        w = 2.0 * math.pi * k / p
        idx = 4 * (k - 1)

        # u0, u1 give 2 + 2*cos(w*(va + vb))
        W1[idx + 0, :p] = np.cos(w * v)
        W1[idx + 0, p:] = np.cos(w * v)

        W1[idx + 1, :p] = np.sin(w * v)
        W1[idx + 1, p:] = -np.sin(w * v)

        # u2, u3 give 2 + 2*sin(w*(va + vb))
        W1[idx + 2, :p] = np.cos(w * v - math.pi / 4.0)
        W1[idx + 2, p:] = np.cos(w * v - math.pi / 4.0)

        W1[idx + 3, :p] = np.sin(w * v - math.pi / 4.0)
        W1[idx + 3, p:] = -np.sin(w * v - math.pi / 4.0)

        # Output weights for symbol c:
        # cos(w*(va + vb - vc)) = cos(w*(va+vb))*cos(w*vc) + sin(w*(va+vb))*sin(w*vc)
        cos_vc = np.cos(w * v)
        sin_vc = np.sin(w * v)

        W2[:, idx + 0] = cos_vc
        W2[:, idx + 1] = cos_vc
        W2[:, idx + 2] = sin_vc
        W2[:, idx + 3] = sin_vc

    # Also add a constant neuron u_const = 1 (since x has exactly one 1 in first half and one 1 in second half)
    # to subtract the 2*(cos(w*vc) + sin(w*vc)) constant offsets so every target logit is identical.
    # For any pair (a, b), sum(x[:p]) = 1, so W1[224, :p] = 1 gives h[224] = 1, act[224] = 1.
    W1[224, :p] = 1.0
    offset = np.zeros(p, dtype=np.float64)
    for k in range(1, 57):
        w = 2.0 * math.pi * k / p
        offset += 2.0 * (np.cos(w * v) + np.sin(w * v))
    W2[:, 224] = -offset

    return {"W1": W1, "W2": W2}
