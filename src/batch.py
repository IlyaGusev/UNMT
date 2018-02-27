from typing import List, Tuple

import torch
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
        pairs = sorted(zip(src_batch.variable.data, tgt_batch.variable.data), key=lambda s: len(s[0]), reverse=True)
        src_sequences, tgt_sequences = zip(*pairs)
        src_variable = BatchGenerator.get_variable(src_sequences)
        tgt_variable = BatchGenerator.get_variable(tgt_sequences)
        src_lengths = BatchGenerator.get_lengths(src_sequences)
        tgt_lengths = BatchGenerator.get_lengths(tgt_sequences)
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
        return Variable(torch.LongTensor(sequences), requires_grad=False)

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
        src_file_names, tgt_file_names = zip(*self.pair_file_names)
        src_batch_generator = BatchGenerator(src_file_names, self.batch_size, self.max_len,
                                             self.vocabulary, self.languages[0], False, self.max_batch_count)
        tgt_batch_generator = BatchGenerator(tgt_file_names, self.batch_size, self.max_len,
                                             self.vocabulary, self.languages[1], False, self.max_batch_count)
        for src_batch, tgt_batch in zip(src_batch_generator, tgt_batch_generator):
            yield Batch.sort_pair(src_batch, tgt_batch)
