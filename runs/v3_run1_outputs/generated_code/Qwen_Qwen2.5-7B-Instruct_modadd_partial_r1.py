import numpy as np
from math import pi

def build_weights():
    def trig_feature(i):
        return np.array([np.sin(2 * pi * i / 113), np.cos(2 * pi * i / 113)])

    W1 = np.zeros((256, 226))
    for j in range(226):
        a, b = j // 113, j % 113
        W1[:, j] = trig_feature(a) + trig_feature(b)

    W2 = np.zeros((113, 256))
    for i in range(113):
        W2[i, :] = trig_feature(i)

    return {"W1": W1, "W2": W2}