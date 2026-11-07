def build_weights():
    import numpy as np, math
    rng = np.random.RandomState(0)

    # ---------------- conv1 ----------------
    w1 = np.zeros((32, 3, 5, 5), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    ys, xs = np.mgrid[-2:3, -2:3].astype(np.float32)
    r2 = xs**2 + ys**2

    def gauss(s):
        g = np.exp(-r2 / (2 * s * s))
        return g / g.sum()

    colors = [
        (1, 1, 1),        # 0 white
        (-1, -1, -1),     # 1 black
        (1, -.5, -.5),    # 2 red
        (-1, .5, .5),     # 3 cyan
        (-.5, 1, -.5),    # 4 green
        (.5, -1, .5),     # 5 magenta
        (-.5, -.5, 1),    # 6 blue
        (.5, .5, -1),     # 7 yellow
        (.6, .0, -.6),    # 8 brown/orange-ish
        (-.3, .1, 1),     # 9 sky blue
        (-.8, -.3, -.8),  # 10 dark green
        (1, .5, -.2),     # 11 orange/skin
    ]
    g = gauss(1.4)
    for i, c in enumerate(colors):
        c = np.array(c, dtype=np.float32)
        c = c / np.abs(c).sum() * 1.5
        for k in range(3):
            w1[i, k] = c[k] * g

    def gabor(theta, odd=True, sign=1.0):
        u = xs * math.cos(theta) + ys * math.sin(theta)
        e = np.exp(-r2 / (2 * 1.5**2))
        if odd:
            f = np.sin(2 * math.pi * u / 5.0) * e
        else:
            f = np.cos(2 * math.pi * u / 5.0) * e
            f = f - e * (f.sum() / e.sum())
        f = sign * f
        return f / np.abs(f).sum()

    gray = np.array([1, 1, 1], dtype=np.float32) / 3.0
    for k in range(4):
        th = k * math.pi / 4
        for ch, (odd, sg) in zip([12 + k, 16 + k, 20 + k],
                                 [(True, 1.0), (True, -1.0), (False, 1.0)]):
            f = gabor(th, odd, sg)
            for c in range(3):
                w1[ch, c] = gray[c] * f * 3.0
    # color-opponent edges
    opp = [(1, -1, 0), (1, -1, 0), (.5, .5, -1), (.5, .5, -1)]
    ths = [0, math.pi / 2, 0, math.pi / 2]
    for j in range(4):
        f = gabor(ths[j], True, 1.0)
        c = np.array(opp[j], dtype=np.float32)
        for k in range(3):
            w1[24 + j, k] = c[k] * f * 1.5
    # center-surround
    dog = gauss(0.8) - gauss(1.8)
    dog = dog / np.abs(dog).sum() * 2
    for k in range(3):
        w1[28, k] = dog / 3 * 3
        w1[29, k] = -dog / 3 * 3
    cr = np.array([1, -.5, -.5], dtype=np.float32)
    cb = np.array([-.5, -.5, 1], dtype=np.float32)
    for k in range(3):
        w1[30, k] = cr[k] * dog * 1.5
        w1[31, k] = cb[k] * dog * 1.5
    b1[:] = 0.0
    b1[:12] = -0.1

    # ---------------- conv2 ----------------
    w2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    for i in range(32):
        w2[i, i] += 0.0625
        w2[i, i, 1, 1] += 0.5
    for j in range(32):
        a = j % 12
        b = 12 + (j * 5) % 16
        w2[32 + j, a] += 0.5 / 9.0
        w2[32 + j, b] += 0.5 / 9.0
        b2[32 + j] = -0.05
    w2[32:] += 0.003 * rng.randn(32, 32, 3, 3).astype(np.float32)

    # ---------------- fc ----------------
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']
    P = np.zeros((10, 12), dtype=np.float32)
    def setp(c, d):
        for k, v in d.items():
            P[c, k] = v
    # color idx: 0 white,1 black,2 red,3 cyan,4 green,5 magenta,6 blue,7 yellow,8 brown,9 sky,10 dkgreen,11 orange
    setp(0, {0: .5, 9: 1.0, 6: .5, 1: -.3, 4: -.5, 8: -.5, 10: -.4})
    setp(1, {2: .6, 1: .3, 0: .2, 6: .1, 4: -.2})
    setp(2, {4: .2, 8: .3, 9: .3, 0: .1})
    setp(3, {8: .4, 1: .2, 0: .2, 11: .2, 6: -.2})
    setp(4, {8: .6, 4: .5, 10: .3, 9: -.2, 6: -.3})
    setp(5, {8: .5, 0: .2, 11: .3, 6: -.2})
    setp(6, {4: .8, 10: .8, 8: .2, 1: .2, 6: -.4, 9: -.4})
    setp(7, {8: .6, 4: .4, 9: .2, 10: .2, 6: -.2})
    setp(8, {6: .7, 9: .5, 0: .3, 3: .2, 8: -.4, 4: -.3})
    setp(9, {2: .3, 0: .3, 1: .3, 6: .2, 4: -.2})

    row = np.ones((10, 4), dtype=np.float32)
    row[0] = [1.6, 1.2, .7, .5]
    row[8] = [.6, .8, 1.3, 1.4]
    for c in (4, 6, 7):
        row[c] = [.6, .9, 1.2, 1.4]
    row[1] = [.6, .9, 1.3, 1.2]
    row[9] = [.6, .9, 1.3, 1.2]

    E = np.array([-.3, .5, .1, .15, .1, .15, .25, .1, -.2, .5], dtype=np.float32)
    ob = np.zeros((10, 4), dtype=np.float32)
    ob[1] = [.4, 0, .4, 0]
    ob[9] = [.4, 0, .5, 0]
    ob[8] = [.5, 0, -.2, 0]
    ob[0] = [.2, 0, -.2, 0]
    ob[4] = [-.1, 0, .4, 0]
    ob[7] = [-.1, 0, .4, 0]

    def edge_prior(c, b):
        o = b - 12
        if o < 16:
            return E[c] * 0.5 + ob[c, o % 4]
        return E[c] * 0.3

    W = np.zeros((10, 64, 4, 4), dtype=np.float32)
    for c in range(10):
        for y in range(4):
            r = row[c, y]
            for i in range(12):
                W[c, i, y, :] = P[c, i] * r * 0.08
            for b in range(12, 28):
                W[c, b, y, :] = edge_prior(c, b) * 0.04 * (1 if (b - 12) < 8 else 0.5)
            # center-surround / blob channels: mild texture prior
            W[c, 28:30, y, :] = E[c] * 0.03
            W[c, 30, y, :] = P[c, 2] * r * 0.05
            W[c, 31, y, :] = P[c, 6] * r * 0.05
            for j in range(32):
                a = j % 12
                b = 12 + (j * 5) % 16
                W[c, 32 + j, y, :] = P[c, a] * np.sign(edge_prior(c, b) + 0.05) * r * 0.05 \
                    + edge_prior(c, b) * 0.02
    fcw = W.reshape(10, 1024).astype(np.float32)
    fcb = np.zeros(10, dtype=np.float32)

    return {
        "conv1.weight": w1, "conv1.bias": b1,
        "conv2.weight": w2, "conv2.bias": b2,
        "fc.weight": fcw, "fc.bias": fcb,
    }