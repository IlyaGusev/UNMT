# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Batch creation.

from typing import List, Tuple

import torch
import copy
from torch.autograd import Variable

from utils.vocabulary import Vocabulary


class Batch:
    def __init__(self, variable: Variable, lengths: List[int]):
        self.variable = variable
        self.lengths = lengths

    def cuda(self):
        return Batch(self.variable.cuda(), self.lengths)

    def __str__(self):
        return "Batch: " + str(self.variable) + ", " + str(self.lengths)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def sort_pair(src_batch: 'Batch', tgt_batch: 'Batch') -> Tuple['Batch', 'Batch']:
        batch_size = src_batch.variable.size(1)
        src_data = src_batch.variable.transpose(0, 1).data.cpu()
        tgt_data = tgt_batch.variable.transpose(0, 1).data.cpu()
        tuples = sorted([(list(src_data[b]), list(tgt_data[b]), src_batch.lengths[b], tgt_batch.lengths[b]) 
                         for b in range(batch_size)],
                        key=lambda t: t[2], reverse=True)
        src_sequences = [copy.deepcopy(t[0]) for t in tuples]
        tgt_sequences = [copy.deepcopy(t[1]) for t in tuples]
        src_lengths = [copy.deepcopy(t[2]) for t in tuples]
        tgt_lengths = [copy.deepcopy(t[3]) for t in tuples]
        src_variable = BatchGenerator.get_variable(src_sequences)
        tgt_variable = BatchGenerator.get_variable(tgt_sequences)
        return Batch(src_variable, src_lengths), Batch(tgt_variable, tgt_lengths)


class BatchGenerator:
    def __init__(self, file_names: List[str], batch_size: int, max_len: int,
                 vocabulary: Vocabulary, language: str, is_sorting=True, max_batch_count: int=None):
        self.file_names = file_names  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.max_len = max_len  # type: int
        self.vocabulary = vocabulary  # type: Vocabulary
        self.language = language  # type: str
        self.is_sorting = is_sorting  # type: bool
        self.max_batch_count = max_batch_count  # type: int

    def __iter__(self):
        batch_count = 0
        for file_name in self.file_names:
            seqs = []
            with open(file_name, "r", encoding='utf-8') as r:
                for sentence in r:
                    indices = self.vocabulary.get_indices(sentence, self.language)
                    if len(indices) > self.max_len:
                        continue
                    seqs.append(indices)
                    if len(seqs) == self.batch_size:
                        yield self.process(seqs)
                        batch_count += 1
                        if self.max_batch_count is not None and batch_count == self.max_batch_count:
                            return
                        seqs = []
            if len(seqs) == self.batch_size:
                yield self.process(seqs)

    def process(self, sequences: List[List[int]]):
        if self.is_sorting:
            sequences = self.sort(sequences)
        lengths = self.get_lengths(sequences)
        sequences = self.pad(sequences, lengths)
        variable = self.get_variable(sequences)
        return Batch(variable, lengths)

    @staticmethod
    def sort(sequences: List[List[int]]):
        return list(sorted(sequences, key=lambda s: len(s), reverse=True))

    @staticmethod
    def get_lengths(sequences: List[List[int]]):
        return [len(indices) for indices in sequences]

    @staticmethod
    def get_variable(sequences: List[List[int]]):
        return Variable(torch.LongTensor(sequences), requires_grad=False).transpose(0, 1)

    def pad(self, sequences: List[List[int]], lengths: List[int]):
        return [self.vocabulary.pad_indices(indices, max(lengths), self.language) for indices in sequences]


class BilingualBatchGenerator:
    def __init__(self, pair_file_names: List[Tuple[str, str]], batch_size: int, max_len: int,
                 vocabulary: Vocabulary, languages: List[str], max_batch_count: int=None):
        self.pair_file_names = pair_file_names  # type: List[Tuple[str, str]]
        self.batch_size = batch_size  # type: int
        self.max_len = max_len  # type: int
        self.vocabulary = vocabulary  # type: Vocabulary
        self.languages = languages  # type: List[str]
        self.max_batch_count = max_batch_count  # type: int
        
    def __iter__(self):
        batch_count = 0
        for src_file_name, tgt_file_name in self.pair_file_names:
            src_seqs = []
            tgt_seqs = []
            with open(src_file_name, "r", encoding='utf-8') as src, open(tgt_file_name, "r", encoding='utf-8') as tgt:
                for src_sentence, tgt_sentence in zip(src, tgt):
                    src_indices = self.vocabulary.get_indices(src_sentence, self.languages[0])
                    tgt_indices = self.vocabulary.get_indices(tgt_sentence, self.languages[1])
                    if len(src_indices) > self.max_len or len(tgt_indices) > self.max_len:
                        continue
                    src_seqs.append(src_indices)
                    tgt_seqs.append(tgt_indices)
                    if len(src_seqs) == self.batch_size:
                        yield Batch.sort_pair(self.process(src_seqs, self.languages[0]),
                                              self.process(tgt_seqs, self.languages[1]))
                        batch_count += 1
                        if self.max_batch_count is not None and batch_count == self.max_batch_count:
                            return
                        src_seqs = []
                        tgt_seqs = []
            if len(src_seqs) == self.batch_size:
                yield Batch.sort_pair(self.process(src_seqs, self.languages[0]),
                                      self.process(tgt_seqs, self.languages[1]))
    
    def process(self, sequences: List[List[int]], language: str):
        lengths = BatchGenerator.get_lengths(sequences)
        sequences = self.pad(sequences, lengths, language)
        variable = BatchGenerator.get_variable(sequences)
        return Batch(variable, lengths)
    
    def pad(self, sequences: List[List[int]], lengths: List[int], language: str):
        return [self.vocabulary.pad_indices(indices, max(lengths), language) for indices in sequences]
