# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Word-by-word dictionary-based translation model.

from typing import List, Dict, Set

import torch
from torch.autograd import Variable

from utils.vocabulary import Vocabulary
from src.translator import TranslationModel
from collections import defaultdict
import random


class WordByWordModel(TranslationModel):
    def __init__(self, src_to_tgt_dict_filename: str, tgt_to_src_dict_filename: str,
                 all_vocabulary: Vocabulary, max_length: int):
        self.max_length = max_length
        self.src_to_tgt_dict_filename, self.tgt_to_src_dict_filename = \
            src_to_tgt_dict_filename, tgt_to_src_dict_filename
        self.all_vocabulary = all_vocabulary

        self.src2tgt = self.init_mapping(src_to_tgt_dict_filename, self.all_vocabulary, "src", "tgt")
        self.src2tgt[self.all_vocabulary.get_pad("src")] = {self.all_vocabulary.get_pad("tgt")}
        self.src2tgt[self.all_vocabulary.get_eos("src")] = {self.all_vocabulary.get_eos("tgt")}
        self.src2tgt[self.all_vocabulary.get_sos("src")] = {self.all_vocabulary.get_sos("tgt")}
        self.src2tgt[self.all_vocabulary.get_unk("src")] = {self.all_vocabulary.get_unk("tgt")}

        self.tgt2src = self.init_mapping(tgt_to_src_dict_filename, self.all_vocabulary, "tgt", "src")
        self.tgt2src[self.all_vocabulary.get_pad("tgt")] = {self.all_vocabulary.get_pad("src")}
        self.tgt2src[self.all_vocabulary.get_eos("tgt")] = {self.all_vocabulary.get_eos("src")}
        self.tgt2src[self.all_vocabulary.get_sos("tgt")] = {self.all_vocabulary.get_sos("src")}
        self.tgt2src[self.all_vocabulary.get_unk("tgt")] = {self.all_vocabulary.get_unk("src")}

    @staticmethod
    def init_mapping(bi_dict_filename: str, vocabulary: Vocabulary, first_lang, second_lang):
        mapping = defaultdict(set)
        with open(bi_dict_filename, "r", encoding='utf-8') as r:
            for line in r:
                first_word, second_word = line.strip().split()
                
                first_index = vocabulary.get_unk(first_lang)
                if vocabulary.has_word(first_word, first_lang):
                    first_index = vocabulary.get_index(first_word, first_lang)
                elif vocabulary.has_word(first_word.capitalize(), first_lang):
                    first_index = vocabulary.get_index(first_word.capitalize(), first_lang)
                
                second_index = vocabulary.get_unk(second_lang)
                if vocabulary.has_word(second_word, second_lang):
                    second_index = vocabulary.get_index(second_word, second_lang)
                elif vocabulary.has_word(second_word.capitalize(), second_lang):
                    second_index = vocabulary.get_index(second_word.capitalize(), second_lang)
               
                mapping[first_index].add(second_index)
        return mapping
    

    def translate_to_tgt(self, variable: Variable, lengths: int):
        return self._map_variable(variable, self.src2tgt, "tgt")

    def translate_to_src(self, variable: Variable, lengths: int):
        return self._map_variable(variable, self.tgt2src, "src")

    def translate_sentence(self, sentence: str, from_lang: str, to_lang: str):
        indices = self.all_vocabulary.get_indices(sentence, from_lang)
        variable = self._indices_to_variable(indices)
        translator = self.translate_to_src if from_lang == "tgt" else self.translate_to_tgt
        output_variable = translator(variable, None)
        output_variable = output_variable.transpose(0, 1)
        output_indices = [i for i in list(output_variable[0].data) if i != 0]
        result = []
        for i, index in enumerate(output_indices):
            unk_index = self.all_vocabulary.get_unk(to_lang)
            if index != unk_index:
                result.append(self.all_vocabulary.get_word(index))
            else:
                # word = self.all_vocabulary.get_word(indices[i])
                # lang_word = to_lang + "-" + word
                # if lang_word in self.all_vocabulary.word2index:
                #     result.append(word)
                # else:
                result.append(self.all_vocabulary.get_word(self.all_vocabulary.get_unk(to_lang)))
        return " ".join(result[:-1])

    def _map_variable(self, variable: Variable, mapping: Dict[int, Set[int]], lang):
        input_max_length = variable.size(0)
        batch_size = variable.size(1)
        output_variable = Variable(torch.zeros(input_max_length, batch_size),
                                   requires_grad=False).type(torch.LongTensor)
        for t in range(input_max_length):
            for i in range(batch_size):
                index = variable[t, i].data[0]
                output_variable[t, i] = random.choice(list(mapping[index])) \
                    if mapping[index] else self.all_vocabulary.get_unk(lang)
        return output_variable

    def _indices_to_variable(self, indices: List[int]):
        indices = indices[:self.max_length]
        variable = Variable(torch.zeros(len(indices), 1), requires_grad=False).type(torch.LongTensor)
        variable[:, 0] = torch.LongTensor(indices)
        return variable
