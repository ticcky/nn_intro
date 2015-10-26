#!/bin/bash
set -ex

mkdir -p chunk
wget https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/chunk.txt --continue

python ../data_seq.py chunk.txt --split 0.8 --fout1 chunk/train_dev.txt --fout2 chunk/test.txt
python ../data_seq.py chunk/train_dev.txt --split 0.8 --fout1 chunk/train.txt --fout2 chunk/dev.txt
