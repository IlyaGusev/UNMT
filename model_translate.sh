#! /usr/bin/env bash

python3 translate.py \
    -src_lang en \
    -tgt_lang ru \
    -lang src \
    -model model_supervised.pt \
    -input ../data/input.tok.tc.txt \
    -output ../data/pred.txt \
    -src_vocabulary ../data/src.pickle \
    -tgt_vocabulary ../data/tgt.pickle \
    -all_vocabulary ../data/all.pickle 