import chainer
import chainer.functions as F
from chainer.optimizers import SGD
import numpy as np
import sklearn.metrics
import time


class LogisticRegression(object):
    def __init__(self, n_features):
        self.model = chainer.FunctionSet(
            W=F.Linear(n_features, 2)
        )

        for param in self.model.parameters:
            param[:] = np.random.uniform(-0.1, 0.1, param.shape)

        self.optimizer = SGD()
        self.optimizer.setup(self.model)

    def forward_loss(self, x, y, train=True):
        x = chainer.Variable(x, volatile=not train)
        y = chainer.Variable(y, volatile=not train)

        loss = F.softmax_cross_entropy(self.model.W(x), y)
        return loss, loss.creator.y

    def learn(self, x, y):
        self.optimizer.zero_grads()

        loss, y_hat = self.forward_loss(x, y, train=True)

        loss.backward()

        self.optimizer.update()

        return loss.data

    def eval(self, mb_x, mb_y):
        mb_y_hat = self.predict(mb_x)

        acc =  sklearn.metrics.accuracy_score(mb_y, mb_y_hat)
        prec = sklearn.metrics.precision_score(mb_y, mb_y_hat)
        recall = sklearn.metrics.recall_score(mb_y, mb_y_hat)

        return acc, prec, recall

    def predict(self, x):
        _, y_hat = self.forward_loss(x, np.zeros((len(x), ), dtype='int32'))

        return np.argmax(y_hat, axis=1)



def main(mb_size=1):
    from data_xor import generate_xor
    from data_lin import generate_lin

    generate_data = generate_lin

    data_train = generate_data(1000, noise=0.0)
    x_train = data_train[:, 1:]
    y_train = np.array(data_train[:, 0], dtype='int32')

    data_test = generate_data(100, noise=0.0)
    x_test = data_test[:, 1:]
    y_test = np.array(data_test[:, 0], dtype='int32')

    lr = LogisticRegression(n_features=2)
    lr.optimizer.lr = 0.2

    loss = 0.0
    n_instances = 0
    begin_t = last_print_t = time.time()

    while n_instances < len(data_train) * 10:
        i = np.random.randint(0, len(data_train))
        curr_loss = lr.learn(x_train[i:i + mb_size], y_train[i:i + mb_size])
        loss = loss * n_instances / (n_instances + 1) + 1.0 / (n_instances + 1) * curr_loss

        n_instances += 1

        if time.time() - last_print_t > 1.0:
            last_print_t = time.time()
            print '> t(%.1f) train_loss(%.3f) examples(%d)' % (last_print_t - begin_t, loss, n_instances, )

    a, p, r = lr.eval(x_test, y_test)
    print '# acc(%.3f) prec(%.3f) recall(%.3f)' % (a, p, r,)







if __name__ == '__main__':
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))