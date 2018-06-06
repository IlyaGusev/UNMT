# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Vocabulary with save/load support.

from collections import Counter
from typing import List, Tuple
import pickle
import logging
import os


class Vocabulary:
    def __init__(self, languages: List[str]):
        self.languages = languages
        self.index2word = list()
        self.word2index = dict()
        self.word2count = Counter()
        self.reset()

    def get_pad(self, language):
        return self.word2index[language+"-<pad>"]

    def get_sos(self, language):
        return self.word2index[language+"-<sos>"]

    def get_eos(self, language):
        return self.word2index[language+"-<eos>"]

    def get_unk(self, language):
        return self.word2index[language+"-<unk>"]

    def add_sentence(self, sentence, language):
        for word in sentence.strip().split():
            self.add_word(word, language)

    def add_word(self, word, language):
        word = language+"-"+word
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] += 1
            self.index2word.append(word)
        else:
            self.word2count[word] += 1
    
    def has_word(self, word, language) -> bool:
        word = language+"-"+word
        return word in self.word2index

    def add_file(self, filename: str, language: str):
        with open(filename, "r", encoding="utf-8") as r:
            for line in r:
                for word in line.strip().split():
                    self.add_word(word, language)

    def get_index(self, word, language):
        word = language + "-" + word
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.get_unk(language)

    def get_word(self, index):
        return self.index2word[index].split("-", maxsplit=1)[-1]

    def get_language(self, index):
        return self.index2word[index].split("-", maxsplit=1)[0]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        empty_size = len(self.languages) * 4
        return self.size() <= empty_size

    def shrink(self, n):
        best_words = self.word2count.most_common(n)
        self.reset()
        for word, count in best_words:
            language, word = word.split("-", maxsplit=1)
            self.add_word(word, language)
            self.word2count[word] = count

    def reset(self):
        self.word2count = Counter()
        self.index2word = []
        for language in self.languages:
            self.index2word += [language + "-<pad>", language + "-<sos>",
                                language + "-<eos>", language + "-<unk>"]
        self.word2index = {word: index for index, word in enumerate(self.index2word)}

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    def get_indices(self, sentence: str, language: str) -> List[int]:
        return [self.get_index(word, language) for word in sentence.strip().split()] + [self.get_eos(language)]

    def pad_indices(self, indices: List[int], max_length: int, language: str):
        return indices + [self.get_pad(language) for _ in range(max_length - len(indices))]

    @staticmethod
    def merge(vocab1, vocab2):
        vocabulary = Vocabulary(languages=vocab1.languages + vocab2.languages)
        for i in range(vocab1.size()):
            language = vocab1.get_language(i)
            word = vocab1.get_word(i)
            vocabulary.add_word(word, language)
        for i in range(vocab2.size()):
            language = vocab2.get_language(i)
            word = vocab2.get_word(i)
            vocabulary.add_word(word, language)
        return vocabulary


def collect_vocabularies(src_vocabulary_path: str, tgt_vocabulary_path: str, all_vocabulary_path: str,
                         src_file_names: Tuple[str]=(), tgt_file_names: Tuple[str]=(), src_max_words: int=40000,
                         tgt_max_words: int=40000, reset: bool=True):
    logging.info("Collecting vocabulary...")
    src_vocabulary = Vocabulary(languages=["src"])
    tgt_vocabulary = Vocabulary(languages=["tgt"])
    vocabulary = Vocabulary(languages=["src", "tgt"])

    if not reset and os.path.exists(src_vocabulary_path):
        src_vocabulary.load(src_vocabulary_path)
        tgt_vocabulary.load(tgt_vocabulary_path)
        vocabulary.load(all_vocabulary_path)
        return src_vocabulary, tgt_vocabulary, vocabulary

    assert len(src_file_names) != 0 and len(tgt_file_names) != 0

    src_vocabulary.reset()
    tgt_vocabulary.reset()
    vocabulary.reset()

    for filename in src_file_names:
        src_vocabulary.add_file(filename, "src")
    src_vocabulary.shrink(src_max_words)
    src_vocabulary.save(src_vocabulary_path)

    for filename in tgt_file_names:
        tgt_vocabulary.add_file(filename, "tgt")
    tgt_vocabulary.shrink(tgt_max_words)
    tgt_vocabulary.save(tgt_vocabulary_path)

    vocabulary = Vocabulary.merge(src_vocabulary, tgt_vocabulary)
    vocabulary.save(all_vocabulary_path)
    assert vocabulary.size() == src_vocabulary.size() + tgt_vocabulary.size()
    return src_vocabulary, tgt_vocabulary, vocabulary
