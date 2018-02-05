#! /usr/bin/env bash

python3 train.py \
    -src_lang en \
    -tgt_lang ru \
    -train_src_mono ../data/corpus.tok.clean.tc.en \
    -train_tgt_mono ../data/corpus.tok.clean.tc.ru \
    -train_src_bi ../data/parallel.tok.tc.en \
    -train_tgt_bi ../data/parallel.tok.tc.ru \
    -layers 3 \
    -rnn_size 4 \
    -src_vocab_size 500 \
    -tgt_vocab_size 500 \
    -print_every 5 \
    -save_every 100 \
    -batch_size 8 \
    -supervised_epochs 1 \
    -unsupervised_epochs 1 \
    -n_supervised_batches 100 \
    -n_unsupervised_batches 100 \
    -discriminator_hidden_size 128 \
    -src_vocabulary ../data/src.pickle \
    -tgt_vocabulary ../data/tgt.pickle \
    -all_vocabulary ../data/all.pickle 