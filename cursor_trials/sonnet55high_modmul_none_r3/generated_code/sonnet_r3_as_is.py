def build_weights():
    p = 113
    n = p - 1  # order of the multiplicative group

    # find a primitive root g mod p
    g = None
    for cand in range(2, p):
        if len({pow(cand, e, p) for e in range(n)}) == n:
            g = cand
            break

    # discrete log table: log[g^e mod p] = e
    log = np.zeros(p, dtype=np.int64)
    for e in range(n):
        log[pow(g, e, p)] = e

    vals = np.arange(1, p)      # nonzero residues
    L = log[vals].astype(float) # discrete logs (used for inputs a, b and for class c)

    W1 = np.zeros((256, 2 * p))
    W2 = np.zeros((p, 256))
    ca = vals          # columns of one-hot(a)
    cb = p + vals      # columns of one-hot(b)

    bias = np.zeros(n)  # l-dependent constant produced by the squares

    # Frequencies k = 1..55: four neurons each
    # h1 = cos i + cos j        h2 = -sin i + sin j     -> act1+act2 = 2 + 2cos(w(i+j))
    # h3 = cos i + sin j        h4 =  sin i + cos j     -> act3+act4 = 2 + 2sin(w(i+j))
    for k in range(1, 56):
        w = 2.0 * math.pi * k / n
        r = 4 * (k - 1)
        ci, si = np.cos(w * L), np.sin(w * L)

        W1[r, ca] = ci;      W1[r, cb] = ci
        W1[r + 1, ca] = -si; W1[r + 1, cb] = si
        W1[r + 2, ca] = ci;  W1[r + 2, cb] = si
        W1[r + 3, ca] = si;  W1[r + 3, cb] = ci

        # output: cos(w(S-l)) = cos(wS)cos(wl) + sin(wS)sin(wl)
        W2[1:, r] = 0.5 * ci
        W2[1:, r + 1] = 0.5 * ci
        W2[1:, r + 2] = 0.5 * si
        W2[1:, r + 3] = 0.5 * si

        bias += ci + si

    # Frequency k = 56 (w = pi): one neuron, h = (-1)^i + (-1)^j
    r = 220
    sg = np.cos(math.pi * L)
    W1[r, ca] = sg
    W1[r, cb] = sg
    W2[1:, r] = 0.5 * sg
    bias += sg

    # Constant neuron: h = 1 for every valid input (a in 1..112), act = 1.
    # It cancels the class-dependent constant left over from the squares.
    r = 221
    W1[r, ca] = 1.0
    W2[1:, r] = -bias[log[vals]]

    # Resulting logit for class c (l = log c), S = log a + log b, d = S - l mod 112:
    #   (112*[d==0] - 1 + (-1)^d) / 2   -> 56 at the correct class, <= 0 elsewhere.
    # Class 0 has all-zero weights (logit 0), so it never wins.
    return {"W1": W1, "W2": W2}