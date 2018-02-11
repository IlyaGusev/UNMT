import torch
from torch.autograd import Variable

from utils.batch import indices_from_sentence
from utils.vocabulary import Vocabulary
from src.unmt import UNMT


class Translator:
    @staticmethod
    def translate(model: UNMT, sentence, src_lang, tgt_lang, vocabulary: Vocabulary, use_cuda):
        model.eval()
        input_batches = dict()
        sos_indices = dict()
        input_batches[src_lang], lengths = Translator.sentence_to_variable(sentence, src_lang, vocabulary, use_cuda)
        sos_indices[src_lang] = vocabulary.get_lang_sos(tgt_lang)
        result = model.forward(input_batches, sos_indices)
        translated = list(result.output_variable[:, 0].cpu().data.numpy())
        words = []
        for i in translated:
            word = vocabulary.get_word_lang(i)
            if word == "</s>" or word == "<pad>":
                break
            words.append(word)
        return " ".join(words)

    @staticmethod
    def sentence_to_variable(sentence, lang, vocabulary, use_cuda):
        indices = indices_from_sentence(sentence, vocabulary, lang)
        variable = Variable(torch.zeros(1, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if use_cuda else variable
        lengths = [len(indices)]
        return variable, lengths
