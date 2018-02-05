from typing import List, Tuple
import numpy as np
from collections import Counter


def shuffle(pair_filenames: List[Tuple[str, str]]):
    for filename1, filename2 in pair_filenames:
        with open(filename1, "r", encoding="utf-8") as r1, open(filename2, "r", encoding="utf-8") as r2:
            lines1 = r1.readlines()
            lines2 = r2.readlines()
            assert len(lines1) == len(lines2)
        np.random.seed(42)
        l = len(lines1)
        perm = np.random.permutation(l)
        perm = perm[:8000000]
        val_part = 0.05
        idx_train = perm[:int(len(perm) * (1 - val_part))]
        idx_val = perm[int(len(perm) * (1 - val_part)):]
        with open("src-train.txt", "w", encoding="utf-8") as w1, \
                open("tgt-train.txt", "w", encoding="utf-8") as w2:
            for index in idx_train:
                w1.write(lines1[index])
                w2.write(lines2[index])
        with open("src-val.txt", "w", encoding="utf-8") as w1, \
                open("tgt-val.txt", "w", encoding="utf-8") as w2:
            for index in idx_val:
                w1.write(lines1[index])
                w2.write(lines2[index])


def count_vocab(filename1, filename2):
    src_vocab = Counter()
    tgt_vocab = Counter()
    with open(filename1, "r", encoding="utf-8") as r1, open(filename2, "r", encoding="utf-8") as r2:
        for line1, line2 in zip(r1, r2):
            for word in line1.split():
                src_vocab[word] += 1
            for word in line2.split():
                tgt_vocab[word] += 1
    print(len([word for word, count in src_vocab.items() if count > 5]))
    print(len([word for word, count in tgt_vocab.items() if count > 5]))

INPUT_FILENAME = "/media/yallen/My Passport/Datasets/MT/train.tok.clean.en"
OUTPUT_FILENAME = "/media/yallen/My Passport/Datasets/MT/train.tok.clean.de"
shuffle([(INPUT_FILENAME, OUTPUT_FILENAME)])
# count_vocab(INPUT_FILENAME, OUTPUT_FILENAME)