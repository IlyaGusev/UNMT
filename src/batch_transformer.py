import numpy as np
import torch
from torch.autograd import Variable
from utils.batch import OneLangBatch


class BatchTransformer:
    @staticmethod
    def translate_with_noise(batch: OneLangBatch, translation_func, drop_probability=0.1, shuffle_max_distance=3):
        new_variable, _ = BatchTransformer.translate_variable(translation_func, batch.variable)
        new_variable, new_lengths, permutation = \
            BatchTransformer.noise_variable(new_variable, drop_probability, shuffle_max_distance)
        new_old_batch = BatchTransformer.get_permutated_batch(batch.variable, batch.lengths, permutation)
        return OneLangBatch(new_variable, new_lengths), new_old_batch

    @staticmethod
    def noise(batch: OneLangBatch, drop_probability=0.1, shuffle_max_distance=3):
        new_variable, new_lengths, permutation = \
            BatchTransformer.noise_variable(batch.variable, drop_probability, shuffle_max_distance)
        new_old_batch = BatchTransformer.get_permutated_batch(batch.variable, batch.lengths, permutation)
        return OneLangBatch(new_variable, new_lengths), new_old_batch

    @staticmethod
    def get_permutated_batch(variable, lengths, permutation):
        batch_size = variable.size(1)
        new_old_variable = Variable(torch.zeros(variable.size(0), variable.size(1)).type(torch.LongTensor))
        new_lengths = []
        for b in range(batch_size):
            new_old_variable[:, b] = variable[:, permutation[b]]
            new_lengths.append(lengths[permutation[b]])
        max_length = max(new_lengths)
        new_old_variable = new_old_variable[:max_length, :]
        return OneLangBatch(new_old_variable, new_lengths)

    @staticmethod
    def noise_variable(variable: Variable, drop_probability=0.1, shuffle_max_distance=3):
        batch_size = variable.size(1)
        max_length = variable.size(0)
        variable = variable.transpose(0, 1)
        new_sentences = []
        for b in range(batch_size):
            noisy = BatchTransformer.add_noise(variable[b].data.cpu().numpy(),
                                               drop_probability=drop_probability,
                                               shuffle_max_distance=shuffle_max_distance)
            noisy = noisy + [0] * (max_length - len(noisy))
            new_sentences.append((b, np.array(noisy)))
        new_sentences = sorted(new_sentences, key=lambda p: len([i for i in p[1] if i != 0]), reverse=True)
        permutation = [original_index for original_index, _ in new_sentences]
        new_sentences = [sentence for _, sentence in new_sentences]

        new_variable = Variable(torch.zeros(batch_size, max_length)).type(torch.LongTensor)
        new_lengths = []
        for b, sentence in enumerate(new_sentences):
            new_variable[b] = torch.LongTensor(sentence)
            new_lengths.append(len([i for i in sentence if i != 0]))
        new_max_length = max(new_lengths)
        new_variable = new_variable[:, :new_max_length]
        return new_variable.transpose(0, 1), new_lengths, permutation

    @staticmethod
    def translate_variable(translation_func, variable: Variable):
        def get_lengths(var):
            batch_size = var.size(1)
            sentences = [[index for index in var[:, b].data if index != 0] for b in range(batch_size)]
            return [len(sentence) for sentence in sentences]

        translated = translation_func(variable=variable, lengths=get_lengths(variable))
        translated.transpose(0, 1)

        lengths = get_lengths(translated)
        max_length = max(lengths)
        new_translated = translated[:max_length, :]

        return new_translated, lengths

    @staticmethod
    def add_noise(sequence, drop_probability=0.1, shuffle_max_distance=3):
        sequence = sequence[sequence > 0]
        reminder = sequence[-2:]
        sequence = sequence[:-2]
        if len(sequence) != 0:
            sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]

            def perm(i):
                return i[0] + (shuffle_max_distance + 1) * np.random.random()
            sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
        sequence = list(sequence) + list(reminder)
        return sequence
