import unittest

import numpy as np
import torch
from src.batch_transformer import BatchTransformer
from torch.autograd import Variable
from utils.batch import OneLangBatch


class TestBatchTransformer(unittest.TestCase):

    def test_prepare_translated_variable(self):
        batch = torch.from_numpy(np.array([
            [997, 1831],
            [51540, 92],
            [3, 26],
            [2770, 21],
            [3, 412],
            [267, 575],
            [42681, 42724],
            [2, 2]
        ], dtype=np.int32))

        translated = np.array([
            [2593, 3, 3, 4864, 3, 680, 3, 2],
            [4311, 237, 60, 37, 1097, 1527, 3, 2]
        ])

        def translation(variable, lengths):
            return Variable(torch.from_numpy(translated.T), requires_grad=False)

        var, lengths = BatchTransformer.prepare_translated_variable(translation, Variable(batch, requires_grad=False))
        np.testing.assert_array_equal(np.array([8, 8]), lengths)
        np.testing.assert_array_equal(translated.T, var.data)

    def test_prepare_noisy_input(self):
        np.random.seed(117)
        batch = torch.from_numpy(np.array([
            [997, 1831, 1831],
            [51540, 92, 92],
            [3, 26, 26],
            [2770, 21, 21],
            [3, 412, 412],
            [267, 575, 2],
            [42681, 2, 0],
            [2, 0, 0]
        ], dtype=np.int32))

        var = Variable(batch, requires_grad=False)
        lengths = [8, 7, 6]
        new_batch, new_old_batch = BatchTransformer.prepare_noisy_input(OneLangBatch(var, lengths), drop_probability=0.3)
        np.testing.assert_equal(np.any(np.not_equal(batch.numpy(), new_old_batch.variable.data)), True)

    def test_add_noise(self):
        res = BatchTransformer.add_noise(np.array([997, 51540, 3, 2770, 3, 267, 42681, 2]))
        assert res[-1] == 2
        assert len(res) > 0


if __name__ == '__main__':
    unittest.main()
