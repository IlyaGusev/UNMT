import argparse
import logging
import sys

import torch
from src.translator import Translator
from utils.vocabulary import collect_vocabularies
from src.serialize import load_model


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
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    use_cuda = torch.cuda.is_available()
    logging.info("Use CUDA: " + str(use_cuda))
  
    _, _, vocabulary = collect_vocabularies(
            src_vocabulary_path=opt.src_vocabulary,
            tgt_vocabulary_path=opt.tgt_vocabulary,
            all_vocabulary_path=opt.all_vocabulary, 
            reset=False)
    model, discriminator, main_optimizer, discriminator_optimizer = load_model(opt.model, use_cuda)
    input_filename = opt.input
    output_filename = opt.output
    lang = opt.lang
    tgt_lang = "src" if lang == "tgt" else "tgt"
    translator = Translator(model, vocabulary, use_cuda)
    logging.info("Writing output...")
    with open(input_filename, "r", encoding="utf-8") as r, open(output_filename, "w", encoding="utf-8") as w:
        for line in r:
            translated = translator.translate_sentence(line, lang, tgt_lang)
            logging.debug(translated)
            w.write(translated+"\n")

if __name__ == "__main__":
    main()
