from typing import List, Tuple

import torch
from torch.autograd import Variable
from utils.tqdm import tqdm_open
from utils.vocabulary import Vocabulary


class OneLangBatch:
    def __init__(self, variable, lengths):
        self.variable = variable
        self.lengths = lengths

    def cuda(self):
        return OneLangBatch(self.variable.cuda(), self.lengths)

    def __str__(self):
        return "OneLangBatch: " + str(self.variable) + ", " + str(self.lengths)

    def __repr__(self):
        return self.__str__()


class BilingualBatch:
    def __init__(self, src_variable, tgt_variable, src_lengths, tgt_lengths):
        self.src_variable = src_variable
        self.tgt_variable = tgt_variable
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths

    def cuda(self):
        return BilingualBatch(self.src_variable.cuda(), self.tgt_variable.cuda(), self.src_lengths, self.tgt_lengths)

    def __str__(self):
        return "BilingualBatch: " + str(self.src_variable) + ", " + str(self.tgt_variable) + ", " + str(
            self.src_lengths) + ", " + str(self.tgt_lengths)


def indices_from_sentence(sentence: str, vocabulary: Vocabulary, lang):
    return [vocabulary.get_lang_index(word, lang) for word in sentence.split(' ')] + \
           [vocabulary.get_lang_eos(lang)]


def pad_seq(seq: List[int], vocabulary: Vocabulary, max_length: int):
    seq += [vocabulary.get_pad() for _ in range(max_length - len(seq))]
    return seq


class OneLangBatchGenerator:
    def __init__(self, filenames: List[str], batch_size: int, max_sentence_len: int, vocabulary: Vocabulary, lang: str):
        self.filenames = filenames  # type: List[str, str]
        self.batch_size = batch_size  # type: int
        self.max_sentence_len = max_sentence_len  # type: int
        self.vocabulary = vocabulary
        self.lang = lang

    def __iter__(self):
        for filename in self.filenames:
            seqs = []
            with tqdm_open(filename, encoding='utf-8') as r:
                for sentence in r:
                    sentence = sentence.strip()
                    sentence = indices_from_sentence(sentence, self.vocabulary, self.lang)
                    if len(sentence) >= self.max_sentence_len - 1 or len(sentence) >= self.max_sentence_len - 1:
                        continue

                    seqs.append(sentence)
                    if len(seqs) == self.batch_size:
                        yield self.__process(seqs)
                        seqs = []
            if len(seqs) == self.batch_size:
                yield self.__process(seqs)

    def __process(self, seqs):
        padded, lengths = self.__pad(seqs)
        variable = self.__to_tensor(padded)
        return OneLangBatch(variable, lengths)

    def __pad(self, seqs):
        seqs = sorted(seqs, key=lambda p: len(p), reverse=True)
        lengths = [len(s) for s in seqs]
        padded = [pad_seq(s, self.vocabulary, max(lengths)) for s in seqs]
        return padded, lengths

    def __to_tensor(self, padded):
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        variable = Variable(torch.LongTensor(padded), requires_grad=False).transpose(0, 1)
        return variable


class BilingualBatchGenerator:
    def __init__(self, pair_filenames: List[Tuple[str, str]], batch_size: int, max_sentence_len: int,
                 all_vocabulary: Vocabulary, lang: str, use_cuda: bool = True):
        self.pair_filenames = pair_filenames  # type: List[Tuple[str, str]]
        self.batch_size = batch_size  # type: int
        self.max_sentence_len = max_sentence_len  # type: int
        self.all_vocabulary = all_vocabulary
        self.use_cuda = use_cuda
        self.lang = lang

    def __iter__(self):
        for lang1_filename, lang2_filename in self.pair_filenames:
            input_seqs = []
            output_seqs = []
            with tqdm_open(lang1_filename, encoding='utf-8') as r1, open(lang2_filename, "r", encoding="utf-8") as r2:
                for input_sentence, output_sentence in zip(r1, r2):
                    input_sentence = input_sentence.strip()
                    output_sentence = output_sentence.strip()

                    second_lang = "tgt" if self.lang == "src" else "src"
                    input_sentence = indices_from_sentence(input_sentence, self.all_vocabulary, lang=self.lang)
                    output_sentence = indices_from_sentence(output_sentence, self.all_vocabulary, lang=second_lang)

                    if len(input_sentence) >= self.max_sentence_len - 1 or len(output_sentence) >= self.max_sentence_len - 1:
                        continue

                    input_seqs.append(input_sentence)
                    output_seqs.append(output_sentence)
                    if len(input_seqs) == self.batch_size:
                        yield self.__process(input_seqs, output_seqs)
                        input_seqs = []
                        output_seqs = []
            if len(input_seqs) == self.batch_size:
                yield self.__process(input_seqs, output_seqs)

    def __process(self, input_seqs, output_seqs):
        input_padded, output_padded, input_lengths, output_lengths = self.__pad(input_seqs, output_seqs)
        input_variable, output_variable = self.__to_tensor(input_padded, output_padded)
        return BilingualBatch(input_variable, output_variable, input_lengths, output_lengths)

    def __pad(self, input_seqs, output_seqs):
        seq_pairs = sorted(zip(input_seqs, output_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, self.all_vocabulary, max(input_lengths)) for s in input_seqs]
        output_lengths = [len(s) for s in target_seqs]
        output_padded = [pad_seq(s, self.all_vocabulary, max(output_lengths)) for s in target_seqs]
        return input_padded, output_padded, input_lengths, output_lengths

    def __to_tensor(self, input_padded, output_padded):
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_variable = Variable(torch.LongTensor(input_padded), requires_grad=False).transpose(0, 1)
        output_variable = Variable(torch.LongTensor(output_padded), requires_grad=False).transpose(0, 1)
        return input_variable, output_variable
