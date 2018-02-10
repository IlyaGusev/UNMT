from src.word_by_word import WordByWordModel
from utils.vocabulary import Vocabulary

SRC_LANG = "en"
TGT_LANG = "de"
SRC_TO_TGT_DICT_FILENAME = "./data/en-de/" + SRC_LANG + "-" + TGT_LANG + ".txt"
TGT_TO_SRC_DICT_FILENAME = "./data/en-de/" + TGT_LANG + "-" + SRC_LANG + ".txt"
SRC_FILENAME = "./data/en-de/val.short." + SRC_LANG

if __name__ == "__main__":
    # src_vocabulary = Vocabulary(language="src", path="src.pickle")
    # tgt_vocabulary = Vocabulary(language="tgt", path="tgt.pickle")
    # with open(SRC_TO_TGT_DICT_FILENAME, "r", encoding="utf-8") as r1:
    #     for line in r1:
    #         src, tgt = line.strip().split()
    #         src_vocabulary.add_word(src)
    #         tgt_vocabulary.add_word(tgt)
    # with open(TGT_TO_SRC_DICT_FILENAME, "r", encoding="utf-8") as r1:
    #     for line in r1:
    #         tgt, src = line.strip().split()
    #         src_vocabulary.add_word(src)
    #         tgt_vocabulary.add_word(tgt)
    # all_vocabulary = Vocabulary.merge(src_vocabulary, tgt_vocabulary, "all.pickle")
    all_vocabulary = Vocabulary(language="all", path="./data/en-de/all.pickle")
    model = WordByWordModel(SRC_TO_TGT_DICT_FILENAME, TGT_TO_SRC_DICT_FILENAME, all_vocabulary)

    OUTPUT_FILENAME = "./data/en-de/pred."+TGT_LANG
    with open(SRC_FILENAME, "r", encoding='utf-8') as r:
        with open(OUTPUT_FILENAME, "w", encoding='utf-8') as w:
            for line in r:
                line = line.strip()
                translation = model.translate_sentence(line, "src")
                w.write(" ".join(translation[:-1]) + "\n")
