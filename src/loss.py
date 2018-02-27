import torch
import torch.nn as nn

from utils.vocabulary import Vocabulary
from src.models import Discriminator, Generator


class DiscriminatorLossCompute:
    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()

    def compute(self, encoder_output, target):
        log_prob = self.discriminator.forward(encoder_output)
        log_prob = log_prob.view(-1)
        adv_loss = self.criterion(log_prob, target)
        return adv_loss


class MainLossCompute:
    def __init__(self, generator: Generator, vocabulary: Vocabulary, use_cuda):
        self.generator = generator

        weight = torch.ones(vocabulary.size())
        weight[vocabulary.get_pad("src")] = 0
        weight[vocabulary.get_pad("tgt")] = 0
        weight = weight.cuda() if use_cuda else weight
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute(self, decoder_output, target):
        scores = self.generator.forward(decoder_output)
        target = target.view(-1)
        loss = self.criterion(scores, target)
        return loss
