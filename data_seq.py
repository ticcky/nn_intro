from collections import Counter
import math
import numpy as np
import random


class Dataset(object):
    def __init__(self, seqs, vocab, label_vocab, word_cntr):
        self.seqs = seqs
        self.vocab = vocab
        self.vocab_rev = {v: k for k, v in self.vocab.iteritems()}
        self.label_vocab = label_vocab
        self.label_vocab_rev = {v: k for k, v in self.label_vocab.iteritems()}
        self.word_cntr = word_cntr

    def iter_strings(self):
        vocab_rev = {val: key for key, val in self.vocab.iteritems()}
        label_vocab_rev = {val: key for key, val in self.label_vocab.iteritems()}

        for x, y in self.seqs:
            xs = [vocab_rev[w] for w in x]
            ys = [label_vocab_rev[l] for l in y]

            yield xs, ys

    def get_n_words(self):
        return len(self.vocab)

    def get_n_classes(self):
        return len(self.label_vocab)

    def shuffle(self):
        random.shuffle(self.seqs)

    def split(self, ratio):
        pivot = int(ratio * len(self.seqs))
        seqs1 = self.seqs[:pivot]
        seqs2 = self.seqs[pivot:]
        d1 = Dataset(seqs1, self.vocab, self.label_vocab, self.word_cntr)
        d2 = Dataset(seqs2, self.vocab, self.label_vocab, self.word_cntr)

        return d1, d2

    def prepare_batches(self, n_seqs_per_batch):
        res = []

        n_batches = int(math.ceil(len(self.seqs) * 1.0 / n_seqs_per_batch))

        for seq_id in range(n_batches):
            batch = self.seqs[seq_id * n_seqs_per_batch:(seq_id + 1) * n_seqs_per_batch]
            res.append(self._batch_to_numpy(batch))

        return res

    def _batch_to_numpy(self, batch):
        max_seq_len = None
        for x, y in batch:
            max_seq_len = max(len(x), max_seq_len)

        res_x = np.zeros((max_seq_len, len(batch)), dtype='int32')
        res_y = np.zeros((max_seq_len, len(batch)), dtype='int32') - 1  # Label -1 means that the label is not valid.

        for i, (x, y) in enumerate(batch):
            res_x[:len(x), i] = x
            res_y[:len(y), i] = y

        return res_x.T, res_y.T

    def make_oovs_by_threshold(self, min_occurrence):
        new_vocab = self._new_vocab()
        vocab_rev = {v: k for k, v in self.vocab.iteritems()}

        oov_words = set()
        for word, cnt in sorted(self.word_cntr.items(), key=lambda x: x[1]):
            if cnt >= min_occurrence and word in self.vocab:
                new_vocab[word] = len(new_vocab)

        new_seqs = []
        for x, y in self.seqs:
            new_x = []
            for word_id in x:
                word = vocab_rev[word_id]

                if word in new_vocab:
                    new_x.append(new_vocab[word])
                else:
                    new_x.append(new_vocab['#OOV'])

            new_seqs.append((new_x, y))

        self.seqs = new_seqs
        self.vocab = new_vocab


    @staticmethod
    def load_from_file(fname, based_on=None, embeddings=None):
        frozen_label_vocab, frozen_vocab, label_vocab, vocab = Dataset._initialize_vocab(based_on, embeddings)

        word_cntr = Counter()
        seqs = []
        with open(fname) as f_in:
            x = []
            y = []
            for ln in f_in:
                if not ln.strip():
                    seqs.append((x, y))
                    x = []
                    y = []
                else:
                    xi, yi = ln.split()
                    xi = xi.lower()
                    word_cntr[xi] += 1
                    if not frozen_vocab:
                        if not xi in vocab:
                            vocab[xi] = len(vocab)
                    else:
                        if not xi in vocab:
                            xi = '#OOV'

                    if not frozen_label_vocab:
                        if not yi in label_vocab:
                            label_vocab[yi] = len(label_vocab)
                    else:
                        if not yi in label_vocab:
                            assert False, 'Unknown label. We do not support OOV labels.'

                    x.append(vocab[xi])
                    y.append(label_vocab[yi])

        res = Dataset(seqs, vocab, label_vocab, word_cntr)
        return res

    @staticmethod
    def _initialize_vocab(based_on, embeddings):
        assert not (based_on and embeddings), 'Cannot use both.'
        if not based_on:
            vocab = Dataset._new_vocab()
            label_vocab = {None: 0}
            if embeddings:
                for word in embeddings:
                    vocab[word] = len(vocab)
                frozen_vocab = True
            else:
                frozen_vocab = False
            frozen_label_vocab = False
        elif based_on:
            vocab = based_on.vocab
            label_vocab = based_on.label_vocab
            frozen_vocab = True
            frozen_label_vocab = True
        else:
            assert False, 'Invalid state.'
        return frozen_label_vocab, frozen_vocab, label_vocab, vocab

    @staticmethod
    def _new_vocab():
        return {None: 0, '#OOV': 1}

    def save_to_file(self, fname):
        vocab_rev = {v: k for k, v in self.vocab.iteritems()}
        label_vocab_rev = {v: k for k, v in self.label_vocab.iteritems()}
        with open(fname, 'w') as f_out:
            for x, y in self.seqs:
                for word, label in zip(x, y):
                    f_out.write("%s %s\n" % (vocab_rev[word], label_vocab_rev[label], ))
                f_out.write("\n")




def main(fname, split, dont_shuffle, fout1, fout2):
    ds = Dataset.load_from_file(fname)
    if not dont_shuffle:
        ds.shuffle()

    ratio = float(split)

    ds1, ds2 = ds.split(ratio)

    ds1.save_to_file(fout1)
    ds2.save_to_file(fout2)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--dont_shuffle', type=bool, default=False)
    parser.add_argument('--fout1')
    parser.add_argument('--fout2')

    args = parser.parse_args()

    main(**vars(args))