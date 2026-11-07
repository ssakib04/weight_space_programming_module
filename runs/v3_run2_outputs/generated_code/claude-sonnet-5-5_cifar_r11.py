def build_weights():
    import numpy as np, math
    r = np.arange(-2, 3, dtype=np.float64)
    X, Y = np.meshgrid(r, r)  # X: column offset, Y: row offset (down)

    def g(s):
        return np.exp(-(X**2 + Y**2) / (2 * s * s))

    def nrm1(a):
        return a / (np.abs(a).sum() + 1e-9)

    def gabor(theta, sign=1.0, lam=4.0, sig=1.3):
        u = Y * math.cos(theta) + X * math.sin(theta)
        return nrm1(sign * np.sin(2 * math.pi * u / lam) * g(sig)) * 2.0

    lum = np.array([0.3, 0.59, 0.11]) * 1.6
    rg = np.array([1.0, -1.0, 0.0]) * 0.7
    by = np.array([-0.5, -0.5, 1.0]) * 0.8
    white = np.array([1.0, 1.0, 1.0]) / 1.7
    blue = np.array([-0.5, -0.3, 1.0])
    green = np.array([-0.5, 1.0, -0.5])
    red = np.array([1.0, -0.5, -0.5])
    brown = np.array([0.6, 0.2, -0.6])
    sky = np.array([0.0, 0.4, 0.9])
    sand = np.array([0.7, 0.7, -0.8])

    blob = nrm1(g(1.2))
    dog = nrm1(g(0.8)) - nrm1(g(2.0))
    fine = nrm1(g(0.6)) - nrm1(g(1.2))
    wide = nrm1(g(2.0))

    F = []  # (colorvec, spatial, bias)
    for o in range(4):
        for p in (0, 1):
            F.append((lum, gabor(o * math.pi / 4, 1.0 if p == 0 else -1.0), 0.0))
    F.append((lum, dog, -0.05))            # 8
    F.append((-lum, dog, -0.05))           # 9
    F.append((rg, blob, -0.05))            # 10
    F.append((-rg, blob, -0.05))           # 11
    F.append((by, blob, -0.05))            # 12
    F.append((-by, blob, -0.05))           # 13
    F.append((white, blob, -0.1))          # 14 white/bright
    F.append((-white, blob, -0.1))         # 15 dark
    F.append((blue, blob, -0.1))           # 16
    F.append((green, blob, -0.1))          # 17
    F.append((red, blob, -0.1))            # 18
    F.append((brown, blob, -0.1))          # 19
    F.append((sky, wide, -0.1))            # 20
    F.append((sand, blob, -0.1))           # 21
    F.append((lum, fine, -0.02))           # 22 fine texture on
    F.append((-lum, fine, -0.02))          # 23 fine texture off
    F.append((green, fine * 1.5, -0.02))   # 24 green texture
    F.append((rg, gabor(0.0), 0.0))        # 25
    F.append((rg, gabor(math.pi / 2), 0.0))  # 26
    F.append((by, gabor(0.0), 0.0))        # 27
    F.append((by, gabor(math.pi / 2), 0.0))  # 28
    F.append((lum, nrm1(-Y * g(1.5)) * 2, 0.0))  # 29 top brighter
    F.append((lum, nrm1(Y * g(1.5)) * 2, 0.0))   # 30 bottom brighter
    F.append((lum, wide, 0.0))             # 31 wide luminance

    w1 = np.zeros((32, 3, 5, 5))
    b1 = np.zeros(32)
    for i, (c, s, b) in enumerate(F):
        w1[i] = c[:, None, None] * s[None]
        b1[i] = b

    # conv2
    w2 = np.zeros((64, 32, 3, 3))
    b2 = np.zeros(64)
    blur = np.array([[.5, 1, .5], [1, 2, 1], [.5, 1, .5]]) / 8.0
    partner = [(7 * j + 5) % 32 for j in range(32)]
    shifts = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for j in range(32):
        w2[j, j] = blur * 2.0
        a, b = j, partner[j]
        w2[32 + j, a] = blur * 1.5
        dy, dx = shifts[j % 4]
        t = np.zeros((3, 3))
        t[1 + dy, 1 + dx] = 1.0
        t[1, 1] += 0.5
        w2[32 + j, b] = t / 3.0
        b2[32 + j] = -0.05

    # classifier head
    H = [0, 1]; V = [4, 5]; D = [2, 3, 6, 7]
    E = H + V + D
    prefs = [dict() for _ in range(10)]

    def setp(c, idxs, v):
        for i in idxs:
            prefs[c][i] = prefs[c].get(i, 0) + v

    # airplane
    setp(0, [20, 14, 16, 31, 12, 29], 1.0); setp(0, E, -0.4); setp(0, [22, 23, 24, 17, 19], -0.5)
    # automobile
    setp(1, E, 0.4); setp(1, [18, 15, 14, 25, 26], 0.5); setp(1, [20, 17], -0.4); setp(1, [8, 9], 0.3)
    # bird
    setp(2, [20, 16, 12, 31], 0.3); setp(2, [19, 17, 22, 23], 0.4); setp(2, D, 0.2)
    # cat
    setp(3, [19, 21, 15, 22, 23, 8, 9], 0.5); setp(3, E, 0.2); setp(3, [20, 16], -0.5)
    # deer
    setp(4, [19, 17, 24, 30], 0.7); setp(4, V, 0.5); setp(4, [20, 14], -0.3)
    # dog
    setp(5, [19, 21, 22, 23, 15], 0.5); setp(5, D, 0.2); setp(5, [16, 20], -0.4)
    # frog
    setp(6, [17, 24, 15, 22, 23, 11], 0.9); setp(6, [20, 14, 16], -0.6)
    # horse
    setp(7, [19, 17, 30, 21], 0.6); setp(7, V, 0.5); setp(7, [20], 0.2); setp(7, [16], -0.3)
    # ship
    setp(8, [16, 12, 20, 14, 27], 0.8); setp(8, H, 0.6); setp(8, [19, 17, 24], -0.5)
    # truck
    setp(9, E, 0.4); setp(9, [15, 18, 14, 25, 26, 8], 0.4); setp(9, [17, 24, 20], -0.4)

    rows = np.array([
        [1.3, 0.9, 0.6, 0.5],   # airplane: top
        [0.4, 0.8, 1.2, 1.0],   # automobile
        [1.0, 1.0, 0.9, 0.7],   # bird
        [0.7, 1.1, 1.1, 0.8],   # cat
        [0.5, 0.9, 1.2, 1.2],   # deer
        [0.7, 1.0, 1.1, 0.9],   # dog
        [0.6, 1.0, 1.2, 1.0],   # frog
        [0.6, 1.0, 1.2, 1.2],   # horse
        [0.8, 1.0, 1.1, 1.3],   # ship
        [0.5, 0.9, 1.2, 1.0],   # truck
    ])
    cols = np.array([0.8, 1.1, 1.1, 0.8])

    P = np.zeros((10, 64))
    for c in range(10):
        for f, v in prefs[c].items():
            P[c, f] += v
        for j in range(32):
            P[c, 32 + j] = 0.25 * (P[c, j] + P[c, partner[j]])

    W = np.zeros((10, 64, 4, 4))
    for c in range(10):
        for ch in range(64):
            W[c, ch] = P[c, ch] * rows[c][:, None] * cols[None, :]
    W = W - W.mean(axis=0, keepdims=True)
    W *= 0.08
    fcw = W.reshape(10, 1024)
    fcb = np.zeros(10)

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fcw.astype(np.float32),
        "fc.bias": fcb.astype(np.float32),
    }