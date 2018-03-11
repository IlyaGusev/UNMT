# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Loss calculation.

import torch
import torch.nn as nn

from utils.vocabulary import Vocabulary
from src.models import Discriminator


class DiscriminatorLossCompute:
    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator
        self.criterion = nn.BCELoss(size_average=False)

    def compute(self, encoder_output, target):
        log_prob = self.discriminator.forward(encoder_output)
        log_prob = log_prob.view(-1)
        adv_loss = self.criterion(log_prob, target)
        return adv_loss


class MainLossCompute:
    def __init__(self, vocabulary: Vocabulary, use_cuda):
        weight = torch.ones(vocabulary.size())
        weight[vocabulary.get_pad("src")] = 0
        weight[vocabulary.get_pad("tgt")] = 0
        weight = weight.cuda() if use_cuda else weight
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute(self, scores, target):
        loss = 0
        for t in range(target.size(0)):
            loss += self.criterion(scores[t], target[t])
        return loss
