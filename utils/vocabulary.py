from collections import Counter
import os
import pickle


class Vocabulary:
    def __init__(self, language, path):
        self.language = language
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = ["<pad>", "</b>", "</s>", "<unk>"]
        if os.path.exists(path):
            self.load(path)

    def get_pad(self):
        return self.index2word.index("<pad>")

    def get_sos(self):
        return self.index2word.index("</b>")

    def get_eos(self):
        return self.index2word.index("</s>")

    def get_unk(self):
        return self.index2word.index("<unk>")

    def get_lang_sos(self, lang):
        return self.word2index[lang + "-</b>"]

    def get_lang_eos(self, lang):
        return self.word2index[lang + "-</s>"]

    def get_lang_unk(self, lang):
        return self.word2index[lang + "-<unk>"]

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if word == '':
                continue
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] += 1
            self.index2word.append(word)
        else:
            self.word2count[word] += 1

    def add_lang_word(self, word, lang):
        self.add_word(lang+"-"+word)

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            if "src-" in word:
                return self.get_index("src-<unk>")
            if "tgt-" in word:
                return self.get_index("tgt-<unk>")
            return self.get_unk()

    def get_lang_index(self, word, lang):
        word = lang + "-" + word
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.get_lang_unk(lang)

    def get_word(self, index):
        return self.index2word[index]

    def get_word_lang(self, index):
        return self.index2word[index][4:]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        if self.language != "all":
            return self.size() <= 4
        return self.size() <= 7

    def shrink(self, n):
        best_words = self.word2count.most_common(n)
        self.index2word = ["<pad>", "</b>", "</s>", "<unk>"]
        self.word2index = {}
        self.word2count = Counter()
        for word, count in best_words:
            self.add_word(word)
            self.word2count[word] = count

    def reset(self):
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = ["<pad>", "</b>", "</s>", "<unk>"]

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    @staticmethod
    def merge(vocab1, vocab2, path):
        vocab = Vocabulary(language="all", path=path)
        vocab.index2word = ["<pad>"]
        vocab.index2word += ["src-" + word for word in vocab1.index2word[1:]]
        vocab.index2word += ["tgt-" + word for word in vocab2.index2word[1:]]
        vocab.word2index = {word: index for index, word in enumerate(vocab.index2word)}
        vocab.word2count = Counter(vocab.index2word)
        return vocab
