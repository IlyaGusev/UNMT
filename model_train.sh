#! /usr/bin/env bash

python3 train.py \
    -src_lang en \
    -tgt_lang ru \
    -train_src_mono ../data/corpus.tok.clean.tc.en \
    -train_tgt_mono ../data/corpus.tok.clean.tc.ru \
    -train_src_bi ../data/parallel.tok.tc.en \
    -train_tgt_bi ../data/parallel.tok.tc.ru \
    -layers 3 \
    -rnn_size 400 \
    -src_vocab_size 40000 \
    -tgt_vocab_size 40000 \
    -print_every 100 \
    -save_every 100 \
    -batch_size 64 \
    -src_embeddings ../data/emb.ft.en.vec  \
    -tgt_embeddings ../data/emb.ft.ru.vec \
    -discriminator_hidden_size 1024 \
    -supervised_epochs 5 \
    -unsupervised_epochs 1 \
    -n_unsupervised_batches 1000 \
    -src_vocabulary ../data/src.pickle \
    -tgt_vocabulary ../data/tgt.pickle \
    -all_vocabulary ../data/all.pickle 