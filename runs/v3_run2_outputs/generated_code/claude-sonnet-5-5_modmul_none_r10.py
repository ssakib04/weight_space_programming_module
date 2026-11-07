def build_weights():
    p = 113
    n = p - 1
    # find primitive root
    def is_prim(g):
        x = 1
        seen = set()
        for _ in range(n):
            x = (x * g) % p
            seen.add(x)
        return len(seen) == n
    g = next(t for t in range(2, p) if is_prim(t))
    log = {}
    x = 1
    for e in range(n):
        log[x] = e
        x = (x * g) % p

    K = 32
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    for k in range(1, K + 1):
        w = 2 * math.pi * k / n
        base = 8 * (k - 1)
        for a in range(1, p):
            i = log[a]
            c, s = math.cos(w * i), math.sin(w * i)
            # a-part columns: index a ; b-part columns: index 113+a
            # (sa, ca for a ; sb, cb for b) with signs per neuron
            # cos group
            W1[base + 0, a] = c;  W1[base + 0, 113 + a] = c      # ca+cb
            W1[base + 1, a] = s;  W1[base + 1, 113 + a] = -s     # sa-sb
            W1[base + 2, a] = c;  W1[base + 2, 113 + a] = -c     # ca-cb
            W1[base + 3, a] = s;  W1[base + 3, 113 + a] = s      # sa+sb
            # sin group
            W1[base + 4, a] = s;  W1[base + 4, 113 + a] = c      # sa+cb
            W1[base + 5, a] = c;  W1[base + 5, 113 + a] = s      # ca+sb
            W1[base + 6, a] = s;  W1[base + 6, 113 + a] = -c     # sa-cb
            W1[base + 7, a] = c;  W1[base + 7, 113 + a] = -s     # ca-sb
        for cc in range(1, p):
            l = log[cc]
            co, si = math.cos(w * l), math.sin(w * l)
            W2[cc, base + 0] = co
            W2[cc, base + 1] = co
            W2[cc, base + 2] = -co
            W2[cc, base + 3] = -co
            W2[cc, base + 4] = si
            W2[cc, base + 5] = si
            W2[cc, base + 6] = -si
            W2[cc, base + 7] = -si
    return {"W1": W1, "W2": W2}