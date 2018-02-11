import torch
import torch.nn as nn

from utils.vocabulary import Vocabulary
from src.models import Discriminator, Generator


class DiscriminatorLossCompute:
    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()

    def compute_loss(self, encoder_output, adv_target):
        log_prob = self.discriminator(encoder_output)
        log_prob = log_prob.view(-1)
        adv_loss = self.criterion(log_prob, adv_target)
        return adv_loss, adv_loss.data[0]


class MainLossCompute:
    def __init__(self, generator: Generator, vocab: Vocabulary):
        self.generator = generator
        self.vocab = vocab
        self.padding_idx = vocab.get_pad()

        self.adv_criterion = nn.BCELoss()

        weight = torch.ones(vocab.size())
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute_loss(self, decoder_output, target):
        scores = self.generator(decoder_output)
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        return loss, loss.data[0]
