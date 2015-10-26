import chainer
import chainer.functions as F
from chainer.optimizers import SGD
import numpy as np
import sklearn.metrics


class NeuralNet(object):
    def __init__(self, n_features, n_hidden):
        self.model = chainer.FunctionSet(
            W1=F.Linear(n_features, n_hidden),
            W2=F.Linear(n_hidden, 2),
            activation=F.relu
        )

        for param in self.model.parameters:
            param[:] = np.random.randn(*param.shape)

        self.optimizer = SGD()
        self.optimizer.setup(self.model)

    def forward_loss(self, x, y, train=True):
        x = chainer.Variable(x, volatile=not train)
        y = chainer.Variable(y, volatile=not train)

        h1 = self.model.activation(self.model.W1(x))
        h2 = self.model.W2(h1)

        loss = F.softmax_cross_entropy(h2, y)
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

    def plot_eval(self, mb_x, mb_y):
        pass

def main():
    from train import train

    lr = NeuralNet(n_features=2, n_hidden=10)
    lr.optimizer.lr = 0.2

    train(model=lr, data='lin')
    train(model=lr, data='xor')




if __name__ == '__main__':
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))