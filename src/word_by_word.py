from typing import List, Dict

import torch
from torch.autograd import Variable
from utils.batch import indices_from_sentence
from utils.vocabulary import Vocabulary


class WordByWordModel:
    def __init__(self, src_to_tgt_dict_filename: str, tgt_to_src_dict_filename: str,
                 all_vocabulary: Vocabulary, max_length: int=50):
        self.max_length = max_length
        self.src_to_tgt_dict_filename, self.tgt_to_src_dict_filename = \
            src_to_tgt_dict_filename, tgt_to_src_dict_filename
        self.all_vocabulary = all_vocabulary

        self.src2tgt = self.init_mapping(src_to_tgt_dict_filename, self.all_vocabulary, "src", "tgt")
        self.src2tgt[self.all_vocabulary.get_lang_eos("src")] = self.all_vocabulary.get_lang_eos("tgt")
        self.src2tgt[self.all_vocabulary.get_lang_sos("src")] = self.all_vocabulary.get_lang_sos("tgt")
        self.src2tgt[self.all_vocabulary.get_lang_unk("src")] = self.all_vocabulary.get_lang_unk("tgt")

        self.tgt2src = self.init_mapping(tgt_to_src_dict_filename, self.all_vocabulary, "tgt", "src")
        self.tgt2src[self.all_vocabulary.get_lang_eos("tgt")] = self.all_vocabulary.get_lang_eos("src")
        self.tgt2src[self.all_vocabulary.get_lang_sos("tgt")] = self.all_vocabulary.get_lang_sos("src")
        self.tgt2src[self.all_vocabulary.get_lang_unk("tgt")] = self.all_vocabulary.get_lang_unk("src")

    @staticmethod
    def init_mapping(bi_dict_filename: str, vocabulary: Vocabulary, first_lang, second_lang):
        mapping = {0: 0}
        with open(bi_dict_filename, "r", encoding='utf-8') as r:
            for line in r:
                first_word, second_word = line.strip().split()
                first_index = vocabulary.get_lang_index(first_word, first_lang)
                second_index = vocabulary.get_lang_index(second_word, second_lang)
                mapping[first_index] = second_index
        return mapping

    def translate_src2tgt(self, variable: Variable, lengths: int):
        return self.map_variable(variable, self.src2tgt, "tgt")

    def translate_tgt2src(self, variable: Variable, lengths: int):
        return self.map_variable(variable, self.tgt2src, "src")

    def map_variable(self, variable: Variable, mapping: Dict[int, int], lang):
        input_max_length = variable.size(0)
        batch_size = variable.size(1)
        output_variable = Variable(torch.zeros(input_max_length, batch_size)).type(torch.LongTensor)
        for t in range(input_max_length):
            for i in range(batch_size):
                index = variable[t, i].data[0]
                output_variable[t, i] = mapping[index] if index in mapping else self.all_vocabulary.get_lang_unk(lang)
        return output_variable

    def translate_src2tgt_sentence(self, sentence: str):
        indices = indices_from_sentence(sentence, self.all_vocabulary, "src")
        variable = self.indices_to_variable(indices)
        output_variable = self.translate_src2tgt(variable, None)
        output_variable = output_variable.transpose(0, 1)
        tgt_indices = [i for i in list(output_variable[0].data) if i != 0]
        result = [self.all_vocabulary.get_word_lang(i) for i in tgt_indices]
        return result

    def translate_tgt2src_sentence(self, sentence: str):
        indices = indices_from_sentence(sentence, self.all_vocabulary, "tgt")
        variable = self.indices_to_variable(indices)
        output_variable = self.translate_tgt2src(variable, None)
        output_variable = output_variable.transpose(0, 1)
        src_indices = [i for i in list(output_variable[0].data) if i != 0]
        result = [self.all_vocabulary.get_word_lang(i) for i in src_indices]
        return result

    def indices_to_variable(self, indices: List[int]):
        batch_size = 32
        variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
        indices = indices[:self.max_length]
        for i, index in enumerate(indices):
            variable[i, 0] = index
        for b in range(1, batch_size):
            variable[0, b] = 2
        return variable
