import time
import numpy as np

from data_xor import generate_xor
from data_lin import generate_lin


def train(model, n_epochs=10, data='lin'):
    """Train the given model on the given dataset."""

    data_train, x_test, x_valid, x_train, y_test, y_valid, y_train = _prepare_data(data)
    n_data = len(data_train)

    # Set the learning rate.
    model.optimizer.lr = 0.001  # Good learning rate is around 0.1. We use this one to show the model gradually improves with more iterations.

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
            curr_loss = model.learn(x, y)
            loss += 1.0 / n_data * curr_loss

            n_instances += 1

            # Print something every second so that we keep the frustration low ;)
            if time.time() - last_print_t > 1.0:
                last_print_t = time.time()

                a, p, r = model.eval(x_valid, y_valid)

                #import ipdb; ipdb.set_trace()

                print '> t(%.1f) train_loss(%.3f) examples(%d) valid{acc(%.3f) prec(%.3f) recall(%.3f)}' % (last_print_t - begin_t, loss, n_instances, a, p, r )

    # Compute the metrics and show evaluation on the test set.
    a, p, r = model.eval(x_test, y_test)
    print '# acc(%.3f) prec(%.3f) recall(%.3f)' % (a, p, r,)

    model.plot_eval(x_test, y_test)


def _prepare_data(data):
    """
    :param data: type of data to generate (either lin - for linear 2D separation problem, or 'xor' for XOR problem.
    :return:
    """
    if data == 'lin':
        generate_data = generate_lin
    elif data == 'xor':
        generate_data = generate_xor
    else:
        raise Exception('Unknown dataset: %s' % data)

    data_train = generate_data(1000, noise=0.1)

    x_train = data_train[:, 1:]
    y_train = np.array(data_train[:, 0], dtype='int32')

    data_valid = generate_data(100, noise=0.1)

    x_valid = data_valid[:, 1:]
    y_valid = np.array(data_valid[:, 0], dtype='int32')

    data_test = generate_data(100, noise=0.1)

    x_test = data_test[:, 1:]
    y_test = np.array(data_test[:, 0], dtype='int32')

    return data_train, x_test, x_valid, x_train, y_test, y_valid, y_train