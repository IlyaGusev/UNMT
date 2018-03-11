# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Batch creation.

from typing import List, Tuple
import logging

import torch
import copy
from torch.autograd import Variable

from utils.vocabulary import Vocabulary

logger = logging.getLogger("unmt")


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
    
    def sort(self):
        is_cuda = self.variable.is_cuda
        batch_size = self.variable.size(1)
        data = self.variable.transpose(0, 1).data.cpu()
        tuples = sorted([(list(data[b]), self.lengths[b]) for b in range(batch_size)],
                        key=lambda t: t[1], reverse=True)
        self.lengths = [copy.deepcopy(t[1]) for t in tuples]
        self.variable = BatchGenerator.get_variable([copy.deepcopy(t[0]) for t in tuples])
        self.variable = self.variable.cuda() if is_cuda else self.variable

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
    
    @staticmethod
    def print_pair(batch1, batch2, vocabulary, label):
        data1, data2 = batch1.variable.transpose(0, 1).data.cpu().numpy(), \
                       batch2.variable.transpose(0, 1).data.cpu().numpy()
        unk1 = vocabulary.get_unk("src")
        unk2 = vocabulary.get_unk("tgt")
        for indices1, indices2 in zip(data1, data2):
            sentence1 = " ".join([vocabulary.get_word(index) for index in indices1 if index != unk1 and index != unk2])
            sentence2 = " ".join([vocabulary.get_word(index) for index in indices2 if index != unk1 and index != unk2])
            logger.debug(label + ": " + sentence1 + " --> " + sentence2)


class BatchGenerator:
    def __init__(self, file_names: List[str], num_words_in_batch: int, max_len: int,
                 vocabulary: Vocabulary, language: str, is_sorting=True, max_batch_count: int=None):
        self.file_names = file_names  # type: List[str]
        self.max_len = max_len  # type: int
        self.vocabulary = vocabulary  # type: Vocabulary
        self.language = language  # type: str
        self.is_sorting = is_sorting  # type: bool
        self.max_batch_count = max_batch_count  # type: int
        self.bucket_borders = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 40), (40, 50)]
        self.num_words_in_batch = num_words_in_batch
        self.bucket_batch_size = [num_words_in_batch//(borders[1]-1) for borders in self.bucket_borders]

    def __iter__(self):
        batch_count = 0
        for file_name in self.file_names:
            buckets = [list() for _ in range(len(self.bucket_borders))]
            with open(file_name, "r", encoding='utf-8') as r:
                for sentence in r:
                    indices = self.vocabulary.get_indices(sentence, self.language)
                    if len(indices) > self.max_len:
                        continue
                    for bucket_index, borders in enumerate(self.bucket_borders):
                        if borders[0] <= len(indices) < borders[1]:
                            buckets[bucket_index].append(indices)
                    for bucket_index, bucket in enumerate(buckets):
                        if len(bucket) == self.bucket_batch_size[bucket_index]:
                            yield self.process(bucket)
                            batch_count += 1
                            if self.max_batch_count is not None and batch_count == self.max_batch_count:
                                return
                            buckets[bucket_index] = list()
            for bucket in buckets:
                if len(bucket) == 0:
                    continue
                yield self.process(bucket)
                batch_count += 1
                if self.max_batch_count is not None and batch_count == self.max_batch_count:
                    return

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
    def __init__(self, pair_file_names: List[Tuple[str, str]], max_len: int, num_words_in_batch: int,
                 vocabulary: Vocabulary, languages: List[str], max_batch_count: int=None):
        self.pair_file_names = pair_file_names  # type: List[Tuple[str, str]]
        self.max_len = max_len  # type: int
        self.vocabulary = vocabulary  # type: Vocabulary
        self.languages = languages  # type: List[str]
        self.max_batch_count = max_batch_count  # type: int
        self.bucket_borders = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 40), (40, 50)]
        self.num_words_in_batch = num_words_in_batch
        self.bucket_batch_size = [num_words_in_batch//(borders[1]-1) for borders in self.bucket_borders]
        
    def __iter__(self):
        batch_count = 0
        for src_file_name, tgt_file_name in self.pair_file_names:
            buckets = [list() for _ in range(len(self.bucket_borders))]
            with open(src_file_name, "r", encoding='utf-8') as src, open(tgt_file_name, "r", encoding='utf-8') as tgt:
                for src_sentence, tgt_sentence in zip(src, tgt):
                    src_indices = self.vocabulary.get_indices(src_sentence, self.languages[0])
                    tgt_indices = self.vocabulary.get_indices(tgt_sentence, self.languages[1])
                    if len(src_indices) > self.max_len or len(tgt_indices) > self.max_len or \
                            abs(len(tgt_indices) - len(src_indices)) > 10:
                        continue
                    for bucket_index, borders in enumerate(self.bucket_borders):
                        if borders[0] <= len(src_indices) < borders[1]:
                            buckets[bucket_index].append((src_indices, tgt_indices))
                    for bucket_index, bucket in enumerate(buckets):
                        if len(bucket) == self.bucket_batch_size[bucket_index]:
                            src_seqs, tgt_seqs = zip(*bucket)
                            src_batch, tgt_batch = self.process(src_seqs, self.languages[0]), \
                                                   self.process(tgt_seqs, self.languages[1])
                            Batch.print_pair(src_batch, tgt_batch, self.vocabulary, "initial")
                            yield Batch.sort_pair(src_batch, tgt_batch)
                            batch_count += 1
                            if self.max_batch_count is not None and batch_count == self.max_batch_count:
                                return
                            buckets[bucket_index] = list()
            for bucket in buckets:
                if len(bucket) == 0:
                    continue
                src_seqs, tgt_seqs = zip(*bucket)
                yield Batch.sort_pair(self.process(src_seqs, self.languages[0]),
                                      self.process(tgt_seqs, self.languages[1]))
                batch_count += 1
                if self.max_batch_count is not None and batch_count == self.max_batch_count:
                    return
    
    def process(self, sequences: List[List[int]], language: str):
        lengths = BatchGenerator.get_lengths(sequences)
        sequences = self.pad(sequences, lengths, language)
        variable = BatchGenerator.get_variable(sequences)
        return Batch(variable, lengths)
    
    def pad(self, sequences: List[List[int]], lengths: List[int], language: str):
        return [self.vocabulary.pad_indices(indices, max(lengths), language) for indices in sequences]
