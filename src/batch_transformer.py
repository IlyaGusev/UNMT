# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Operations with batches.

import numpy as np
from typing import Callable, List

import torch
from torch.autograd import Variable

from src.batch import Batch


class BatchTransformer:
    @staticmethod
    def noise(batch: Batch, pad_index: int, drop_probability: float=0.1,
              shuffle_max_distance: int=3) -> Batch:
        new_variable = BatchTransformer.add_noise(batch.variable.transpose(0, 1), pad_index, 
                                                  drop_probability, shuffle_max_distance)
        new_lengths = BatchTransformer.get_lengths(new_variable, pad_index)
        new_variable = new_variable[:, :max(new_lengths)]
        new_batch = Batch(new_variable.transpose(0, 1), new_lengths)
        return new_batch

    @staticmethod
    def translate(batch: Batch, src_pad_index: int, tgt_pad_index: int, tgt_eos_index: int,
                  translation_func: Callable) -> Batch:
        new_variable = translation_func(variable=batch.variable, lengths=batch.lengths)
           
        max_length = new_variable.size(0)
        batch_size = new_variable.size(1)
        for b in range(batch_size):
            sentence = []
            for t in range(max_length):
                word_index = new_variable[t, b].data[0]
                sentence.append(word_index)
                if word_index == tgt_eos_index:
                    break
            if len(sentence) != 0:
                new_variable[:len(sentence), b] = torch.LongTensor(sentence)
            if len(sentence) != max_length:
                new_variable[len(sentence):, b] = torch.add(torch.zeros(max_length-len(sentence)), tgt_pad_index)

        new_lengths = BatchTransformer.get_lengths(new_variable.transpose(0, 1), tgt_pad_index)
        new_variable = new_variable[:max(new_lengths), :]
        assert new_variable.size(1) == batch_size
        return Batch(new_variable, new_lengths)

    @staticmethod
    def add_noise(variable: Variable, pad_index: int, drop_probability: float=0.1,
                  shuffle_max_distance: int=3) -> Variable:
        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()
        
        new_variable = np.zeros((variable.size(0), variable.size(1)), dtype='int')
        variable = variable.data.cpu().numpy()
        for b in range(variable.shape[0]):
            sequence = variable[b]
            sequence = sequence[sequence != pad_index]
            sequence, reminder = sequence[:-1], sequence[-1:]
            if len(sequence) != 0:
                sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]
                sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
            sequence = np.concatenate((sequence, reminder), axis=0)
            sequence = list(np.pad(sequence, (0, variable.shape[1] - len(sequence)), 'constant',
                                   constant_values=pad_index))
            new_variable[b, :] = sequence
        return Variable(torch.LongTensor(new_variable))

    @staticmethod
    def get_lengths(variable: Variable, pad_index: int) -> List[int]:
        return [len(variable[b][variable[b] != pad_index]) for b in range(variable.size(0))]
