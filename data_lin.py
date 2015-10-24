import numpy as np


def generate_lin(n, noise=0.05):
    res = []
    for i in range(n):
        x1 = np.random.uniform(-1.0, 1.0)
        x2 = np.random.uniform(-1.0, 1.0)

        if 0.3 * x1 + 0.9 * x2 + 0.3 > 0:
            y = 0
        else:
            y = 1

        x = np.array([y, x1, x2])
        eps = np.random.randn(2)
        x[1:] += eps * noise

        res.append(x)

    return np.array(res, dtype='float32')


if __name__ == '__main__':
    from utils import vis2d
    data = generate_lin(100)

    vis2d(data)
