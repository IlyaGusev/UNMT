#! /usr/bin/env bash

python3 train.py \
    -src_lang en \
    -tgt_lang de \
    -train_src_mono ./data/en-de/train.mono.tok.clean.tc.en \
    -train_tgt_mono ./data/en-de/train.mono.tok.clean.tc.de \
    -src_embeddings ./data/en-de/wiki.multi.en.vec \
    -tgt_embeddings ./data/en-de/wiki.multi.de.vec \
    -src_vocabulary ./data/en-de/src.pickle \
    -tgt_vocabulary ./data/en-de/tgt.pickle \
    -all_vocabulary ./data/en-de/all.pickle \
    -src_to_tgt_dict ./data/en-de/en-de.txt \
    -tgt_to_src_dict ./data/en-de/de-en.txt \
    -layers 3 \
    -rnn_size 400 \
    -src_vocab_size 40000 \
    -tgt_vocab_size 40000 \
    -print_every 100 \
    -save_every 100 \
    -batch_size 32 \
    -discriminator_hidden_size 1024 \
    -unsupervised_epochs 5 \
    -supervised_epochs 0 \
    -save_model model \
    -reset_vocabularies 0