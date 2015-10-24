import numpy as np


def generate_xor(n, noise=0.1):
    res = []
    for i in range(n):
        x1 = np.random.choice([-1.0, 1.0])
        x2 = np.random.choice([-1.0, 1.0])

        if x1 == x2:
            y = 0
        else:
            y = 1

        x = np.array([y, x1, x2])
        eps = np.random.randn(2) * noise
        x[1:] += eps

        res.append(x)

    return np.array(res, dtype='float32')


if __name__ == '__main__':
    from utils import vis2d
    data = generate_xor(100)

    vis2d(data)
