import chainer
import chainer.functions as F
from chainer.optimizers import SGD
import numpy as np
import sklearn.metrics
import time

import train


class LogisticRegression(object):
    """Logistic regression example in chainer.

    $$ L(x, y) = -log(softmax(Wx + b)_y) $$
    """
    def __init__(self, n_features):
        # Define what parametrized functions the model consists of.
        self.model = chainer.FunctionSet(
            W=F.Linear(n_features, 2)
        )

        # Initialize parameters randomly from gaussian.
        for param in self.model.parameters:
            param[:] = np.random.randn(*param.shape)

        # Define what update rule we will use. SGD is the simplest one,
        #  w' = w + lr * gradient_f(w)
        self.optimizer = SGD()
        self.optimizer.setup(self.model)

    def forward_loss(self, x, y, train=True):
        """Compute the loss function of the model, given the inputs x,
        and labels y.

        Args:
          :arg x Numpy array of dimensionality (batch x input)
          :arg y Numpy array of dimensionality (batch)
        """

        # Wrap the input variables into a class that takes care of remembering
        # the call chain.
        x = chainer.Variable(x, volatile=not train)  # volatile=True means the computation graph will not be
                                                     # built; but for training we need that so we set it to False
        y = chainer.Variable(y, volatile=not train)

        # Apply the functions that define the model.
        wx = self.model.W(x)  # Apply f: f(x) = Wx + b
        loss = F.softmax_cross_entropy(wx, y)  # Apply softmax and crossentropy: f(x, y) = -log(e^{x} / sum(e^{x}))_y

        return loss, loss.creator.y  # loss is an instance of chainer.Variable;
                                     # loss.creator is the computation node that produced the result;
                                     # if you look into the code, it saves softmax outputs as 'y'

    def learn(self, mb_x, mb_y):
        """Update parameters given the training data."""

        self.optimizer.zero_grads()

        # Do the forward pass.
        loss, y_hat = self.forward_loss(mb_x, mb_y, train=True)

        # Do the backward pass from loss (the Jacobian computation).
        loss.backward()

        # Update the parameters W' = W + lr * J^{W}_{loss}(W), b' = b + ...
        self.optimizer.update()

        # Return the "raw" loss (i.e. not chainer.Variable).
        return loss.data

    def eval(self, mb_x, mb_y):
        """Compute some metrics on the given minibatch.
        :param mb_x: Numpy array of float32 of dimensionality (batch x input)
        :param mb_y: Numpy array of int32 of dimensionality (batch) with the labels for each input in mb_x
        :return: Accuracy, Precision, Recall metrics
        """
        mb_y_hat = self.predict(mb_x)  # Get model's predictions about the input data.

        # Compare predictions to the true labels and compute accuracy, precision and recall.
        acc =  sklearn.metrics.accuracy_score(mb_y, mb_y_hat)
        prec = sklearn.metrics.precision_score(mb_y, mb_y_hat)
        recall = sklearn.metrics.recall_score(mb_y, mb_y_hat)

        return acc, prec, recall

    def predict(self, mb_x):
        """Predict labels for the given input minibatch.
        :param mb_x: Numpy array of float32 of dimensionality (batch x input)
        :return: Numpy array of int32 of dimensionality (batch)
        """

        _, y_hat = self.forward_loss(mb_x, np.zeros((len(mb_x), ), dtype='int32'))

        return np.argmax(y_hat, axis=1)

    def plot_eval(self, mb_x, mb_y):
        """Plot the minibatches in 2D and also the separating hyperplane."""
        import matplotlib.pyplot as plt
        import seaborn
        seaborn.set()

        x1 = mb_x[:, 0]
        x2 = mb_x[:, 1]
        y = mb_y

        dec_x1 = np.linspace(-1, 1)

        w1_m_w2 = self.model.W.W[0] - self.model.W.W[1]
        b1_m_b2 = self.model.W.b[0] - self.model.W.b[1]
        dec_x2 = - (w1_m_w2[0] / w1_m_w2[1] * dec_x1) - b1_m_b2 / w1_m_w2[1]

        plt.plot(x1[y == 0], x2[y == 0], 'o', label='Class 0', markersize=3, color='red')
        plt.plot(x1[y == 1], x2[y == 1], 'o', label='Class 1', markersize=3, color='green')
        plt.plot(dec_x1, dec_x2, '-', label='Classifier', color='blue')
        plt.legend()

        plt.show()

    def train(self, n_epochs=10, data='lin'):
        """Train the given model on the given dataset."""

        data_train, x_test, x_valid, x_train, y_test, y_valid, y_train = train._prepare_data(data)
        n_data = len(data_train)

        # Set the learning rate.
        self.optimizer.lr = 0.001  # Good learning rate is around 0.1. We use this one to show the model gradually improves with more iterations.

        n_instances = 0
        begin_t = last_print_t = time.time()

        # Run for the given number of epochs.
        for epoch in range(n_epochs):
            # For SGD it's important to randomize order in which we look at the data points.
            # So for each epoch we randomly choose the order in which we see them.
            order = range(n_data)
            np.random.shuffle(order)

            loss = 0.0
            for i in order:
                x = x_train[i:i + 1]  # We do it this way (instead of x_train[i]) so that the result is of (1 x input) dimensionalit that model.learn expects, instead of just (input).
                y = y_train[i:i + 1]

                # Ask the model to update its parameters given the current example (it uses the model.optimizer rule to update the parameters).
                curr_loss = self.learn(x, y)
                loss += 1.0 / n_data * curr_loss

                n_instances += 1

                # Print something every second so that we keep the frustration low ;)
                if time.time() - last_print_t > 1.0:
                    last_print_t = time.time()

                    a, p, r = self.eval(x_valid, y_valid)

                    #import ipdb; ipdb.set_trace()

                    print '> t(%.1f) train_loss(%.3f) examples(%d) valid{acc(%.3f) prec(%.3f) recall(%.3f)}' % (last_print_t - begin_t, loss, n_instances, a, p, r )

        # Compute the metrics and show evaluation on the test set.
        a, p, r = self.eval(x_test, y_test)
        print '# acc(%.3f) prec(%.3f) recall(%.3f)' % (a, p, r,)

        self.plot_eval(x_test, y_test)


def main():
    lr = LogisticRegression(n_features=2)
    lr.train(data='lin')

    lr = LogisticRegression(n_features=2)
    lr.train(data='xor')




if __name__ == '__main__':
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))