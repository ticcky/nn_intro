import chainer
import chainer.functions as F
from chainer.optimizers import SGD, Adam
import numpy as np
import sklearn.metrics

import os

from data_seq import Dataset


class RNN(object):
    def __init__(self, n_words, emb_size, n_hidden, n_classes, classes):
        self.model = chainer.FunctionSet(
            Emb=F.EmbedID(n_words, emb_size),
            W=F.Linear(emb_size, n_hidden),
            U=F.Linear(n_hidden, n_hidden),
            O=F.Linear(n_hidden, n_classes)
        )

        self.n_hidden = n_hidden
        self.n_clsses = n_classes
        self.emb_size = emb_size

        self.classes = classes
        self.classes_rev = {v: k for k, v in classes.iteritems()}

        for param in self.model.parameters:
            param[:] = np.random.randn(*param.shape) * 0.1

        self.optimizer = Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        self.optimizer.setup(self.model)

    def forward_loss(self, mb_x, mb_y, train=True):
        mb_size = mb_x.shape[0]
        n_steps = mb_x.shape[1]

        loss = 0.0
        h = chainer.Variable(np.zeros((mb_size, self.n_hidden), dtype='float32'), volatile=not train)
        y_hat = []
        for i in range(n_steps):
            x_i = chainer.Variable(mb_x[:, i], volatile=not train)
            y_i = chainer.Variable(mb_y[:, i], volatile=not train)
            h = self.model.W(self.model.Emb(x_i)) + self.model.U(h)
            out = self.model.O(h)

            curr_loss = F.softmax_cross_entropy(out, y_i)
            y_hat.append(curr_loss.creator.y)

            loss += curr_loss * 1.0 / (n_steps * mb_size)

        y_hat = np.array(y_hat).swapaxes(0, 1)

        return loss, y_hat

    def learn(self, x, y):
        self.optimizer.zero_grads()

        loss, y_hat = self.forward_loss(x, y, train=True)

        loss.backward()

        self.optimizer.update()

        return loss.data

    def predict(self, x):
        _, y_hat = self.forward_loss(x, np.zeros(x.shape, dtype='int32'))

        return np.argmax(y_hat, axis=2)

    def predictions_to_text(self, y):
        return [self.classes_rev.get(i, '#EOS') for i in y]

    def eval(self, mb_x, mb_y):
        mb_y_hat = self.predict(mb_x)

        t = self.predictions_to_text

        acc =  sklearn.metrics.accuracy_score(mb_y.flat[mb_y.flat != -1], mb_y_hat.flat[mb_y.flat != -1])
        prec = sklearn.metrics.precision_score(mb_y.flat[mb_y.flat != -1], mb_y_hat.flat[mb_y.flat != -1])
        recall = sklearn.metrics.recall_score(mb_y.flat[mb_y.flat != -1], mb_y_hat.flat[mb_y.flat != -1])
        report = sklearn.metrics.classification_report(t(mb_y.flat[mb_y.flat != -1]), t(mb_y_hat.flat[mb_y.flat != -1]))

        return acc, prec, recall, report, mb_y_hat





def main(data_dir):
    ds_train = Dataset.load_from_file(os.path.join(data_dir, 'train.txt'))
    ds_dev = Dataset.load_from_file(os.path.join(data_dir, 'dev.txt'), based_on=ds_train)
    ds_test = Dataset.load_from_file(os.path.join(data_dir, 'test.txt'), based_on=ds_train)

    batches_train = ds_train.prepare_batches(n_seqs_per_batch=4)
    batches_dev = ds_dev.prepare_batches(n_seqs_per_batch=1000)
    batches_test = ds_test.prepare_batches(n_seqs_per_batch=1000)

    rnn = RNN(n_words=len(ds_train.vocab), emb_size=10, n_hidden=50, n_classes=len(ds_train.label_vocab), classes=ds_train.label_vocab)

    for i in range(1000):
        loss = 0.0
        for mb_x, mb_y in batches_train:
            curr_loss = rnn.learn(mb_x, mb_y)

            loss += curr_loss * 1.0 / len(batches_train)

        a, p, r, report, y_hat = rnn.eval(batches_dev[0][0], batches_dev[0][1])
        print '> loss(%.3f) # dev acc(%.3f) prec(%.3f) recall(%.3f)' % (loss, a, p, r,)

        print report

        print_seqs = zip(batches_dev[0][0], batches_dev[0][1], y_hat)
        #np.random.shuffle(print_seqs)
        for i, (seq_x, seq_y, seq_y_hat) in enumerate(print_seqs[:5]):
            print '# Seq %d (x_i/y_i/yhat_i)' % i
            for x, y, y_hat in zip(seq_x, seq_y, seq_y_hat)[:10]:
                if x == -1 or y == -1:
                    print '--EOS--'
                else:
                    print "%10s %s %5s %5s" % (ds_train.vocab_rev[x], "!" if y != y_hat else " ", ds_train.label_vocab_rev[y], ds_train.label_vocab_rev[y_hat], )
            print

    a, p, r, report, _ = rnn.eval(batches_test[0][0], batches_test[0][1])
    print '#test acc(%.3f) prec(%.3f) recall(%.3f)' % (a, p, r,)




if __name__ == '__main__':
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')

    args = parser.parse_args()

    main(**vars(args))