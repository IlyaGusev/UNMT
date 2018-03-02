# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Translation model. Interchangeable with word-by-word model.

import torch
from torch.autograd import Variable

from src.models import Seq2Seq
from utils.vocabulary import Vocabulary


class TranslationModel:
    def translate_sentence(self, sentence: str, from_lang: str, to_lang: str):
        raise NotImplementedError()

    def translate_to_tgt(self, variable: Variable, lengths: int):
        raise NotImplementedError()

    def translate_to_src(self, variable: Variable, lengths: int):
        raise NotImplementedError()


class Translator(TranslationModel):
    def __init__(self, model: Seq2Seq, vocabulary: Vocabulary, use_cuda: bool):
        self.model = model  # type: Seq2Seq
        self.vocabulary = vocabulary  # type: Vocabulary
        self.use_cuda = use_cuda

    def translate_sentence(self, sentence: str, from_lang: str, to_lang: str):
        variable, lengths = self.sentence_to_variable(sentence, from_lang)
        sos_index = self.vocabulary.get_sos(to_lang)

        output_variable = self.translate(variable, lengths, sos_index)

        translated = list(output_variable[:, 0].cpu().data.numpy())
        words = []
        for i in translated:
            word = self.vocabulary.get_word(i)
            if word == "<eos>" or word == "<pad>":
                break
            words.append(word)
        return " ".join(words)

    def translate(self, variable, lengths, sos_index):
        self.model.eval()
        _, decoder_output = self.model.forward(variable, lengths, sos_index)
        max_length = max(lengths)
        batch_size = variable.size(1)

        output_variable = Variable(torch.zeros(max_length, batch_size).type(torch.LongTensor))
        output_variable = output_variable.cuda() if self.use_cuda else output_variable
        for t in range(max_length):
            output_variable[t] = decoder_output[t].topk(1, dim=1)[1].view(-1)

        output_variable = output_variable.detach()
        return output_variable

    def translate_to_tgt(self, variable: Variable, lengths: int):
        sos_index = self.vocabulary.get_sos("tgt")
        return self.translate(variable, lengths, sos_index)

    def translate_to_src(self, variable: Variable, lengths: int):
        sos_index = self.vocabulary.get_sos("src")
        return self.translate(variable, lengths, sos_index)

    def sentence_to_variable(self, sentence, lang):
        indices = self.vocabulary.get_indices(sentence, lang)[:self.model.max_length]
        variable = Variable(torch.zeros(1, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if self.use_cuda else variable
        lengths = [len(indices)]
        return variable, lengths
