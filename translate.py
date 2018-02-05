import argparse

import torch
from src.trainer import Trainer
from src.translator import Translator


def translate_opts(parser):
    # Languages Options
    group = parser.add_argument_group('Languages')
    group.add_argument('-src_lang', type=str, required=True,
                       help='Src language.')
    group.add_argument('-tgt_lang', type=str, required=True,
                       help='Tgt language.')

    group = parser.add_argument_group('Vocabulary')
    group.add_argument('-src_vocabulary', default="src.pickle",
                       help="Path to src vocab")
    group.add_argument('-tgt_vocabulary', default="tgt.pickle",
                       help="Path to tgt vocab")
    group.add_argument('-all_vocabulary', default="all.pickle",
                       help="Path to all vocab")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')

    group = parser.add_argument_group('Model')
    group.add_argument('-lang', type=str, default="src",
                       help='Src language (src/tgt)')
    group.add_argument('-model', required=True,
                       help='Path to model .pt file')
    group.add_argument('-input',  required=True,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
translate_opts(parser)
opt = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    print("Use CUDA: ", use_cuda)
    state = Trainer(opt.src_lang, opt.tgt_lang, use_cuda=use_cuda)
    state.collect_vocabularies(src_vocabulary_path=opt.src_vocabulary,
                               tgt_vocabulary_path=opt.tgt_vocabulary,
                               all_vocabulary_path=opt.all_vocabulary)
    state.load(opt.model)
    state.model = state.model.cuda() if use_cuda else state.model
    input_filename = opt.input
    output_filename = opt.output
    lang = opt.lang
    tgt_lang = "src" if lang == "tgt" else "tgt"
    print("Writing output...")
    with open(input_filename, "r", encoding="utf-8") as r, open(output_filename, "w", encoding="utf-8") as w:
        for line in r:
            translated = Translator.translate(state.model, line, lang, tgt_lang,
                                              state.all_vocabulary, use_cuda)
            # print(translated)
            w.write(translated+"\n")

if __name__ == "__main__":
    main()
