import argparse
import copy
import sys
import logging

import torch

from src.trainer import Trainer
from src.translator import Translator
from src.models import build_model, load_embeddings
from src.word_by_word import WordByWordModel
from utils.vocabulary import collect_vocabularies
from src.serialize import load_model


def train_opts(parser):
    # Languages Options
    group = parser.add_argument_group('Languages')
    group.add_argument('-src_lang', type=str, required=True,
                       help='Src language.')
    group.add_argument('-tgt_lang', type=str, required=True,
                       help='Tgt language.')

    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-src_vocabulary', default="src.pickle",
                       help="Path to src vocab")
    group.add_argument('-tgt_vocabulary', default="tgt.pickle",
                       help="Path to tgt vocab")
    group.add_argument('-all_vocabulary', default="all.pickle",
                       help="Path to all vocab")
    group.add_argument('-reset_vocabularies', type=int, default=1,
                       help="Reset all vocabularies")
    group.add_argument('-train_src_mono', required=True,
                       help="Path to the training source monolingual data")
    group.add_argument('-train_tgt_mono', required=True,
                       help="Path to the training target monolingual data")
    group.add_argument('-train_src_bi', default=None,
                       help="Path to the training source bilingual data")
    group.add_argument('-train_tgt_bi', default=None,
                       help="Path to the training target bilingual data")
    group.add_argument('-n_unsupervised_batches', type=int, default=None,
                       help="Count of src/tgt batches to process")
    group.add_argument('-n_supervised_batches', type=int, default=None,
                       help="Count of parallel/reverted batches to process")
    group.add_argument('-enable_unsupervised_backtranslation', type=bool, default=False,
                       help="Enable unsupervised backtranslation")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')
    group.add_argument('-enable_embedding_training', type=bool, default=False,
                       help='Enable embedding training.')

    # Zero Model Options
    group = parser.add_argument_group('Zero Model')
    group.add_argument('-src_to_tgt_dict', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_to_src_dict', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')
    group.add_argument('-bootstrapped_model', type=str, default=None,
                       help='Pretrained model used to bootstrap unsupervised learning process.')

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model-Encoder-Decoder')
    group.add_argument('-layers', type=int, default=3,
                       help='Number of layers in enc/dec.')
    group.add_argument('-rnn_size', type=int, default=300,
                       help='Size of rnn hidden states')
    group.add_argument('-discriminator_hidden_size', type=int, default=1024,
                       help='Size of discriminator hidden layers')

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    # Model loading/saving options
    group = parser.add_argument_group('General')
    group.add_argument('-save_model', default='model',
                       help="""Model filename (the model will be saved as
                       <save_model>_epochN_PPL.pt where PPL is the
                       validation perplexity""")
    group.add_argument('-save_every', type=int, default=1000,
                       help='Count of minibatches to save')
    group.add_argument('-seed', type=int, default=1337,
                       help="""Random seed used for the experiments
                       reproducibility.""")
    group.add_argument('-load_from', default=None, type=str,
                       help="Load from file")
    # Logging
    group = parser.add_argument_group('Logging')
    group.add_argument('-print_every', type=int, default=1000,
                       help='Count of minibatches to print')

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('-unsupervised_epochs', type=int, default=2,
                       help='Number of unsupervised training epochs')
    group.add_argument('-supervised_epochs', type=int, default=10,
                       help='Number of supervised training epochs')
    group.add_argument('-adam_beta1', type=float, default=0.5,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=0.0003,
                       help="""Main learning rate.""")
    group.add_argument('-discriminator_lr', type=float, default=0.0005,
                       help="""Discriminator learning rate""")


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
train_opts(parser)
opt = parser.parse_args()


def init_zero_model(vocabulary, use_cuda, src_to_tgt_dict=None, tgt_to_src_dict=None,
                    bootstrapped_model_path=None, train_src_bi=None, train_tgt_bi=None, supervised_epochs=5,
                    batch_size=32, n_supervised_batches=None, print_every=1000, save_every=1000,
                    save_file: str="model", max_length=50):
    if src_to_tgt_dict is not None and tgt_to_src_dict is not None:
        zero_model = WordByWordModel(src_to_tgt_dict_filename=opt.src_to_tgt_dict,
                                     tgt_to_src_dict_filename=opt.tgt_to_src_dict,
                                     all_vocabulary=vocabulary, max_length=max_length)
    elif bootstrapped_model_path is not None:
        zero_model, discriminator, _, _ = load_model(bootstrapped_model_path, use_cuda)
        for param in zero_model.parameters():
            param.requires_grad = False
    elif train_src_bi is not None and train_tgt_bi is not None:
        model, discriminator = build_model(
            max_length=50,
            output_size=vocabulary.size(),
            rnn_size=opt.rnn_size,
            encoder_n_layers=opt.layers,
            decoder_n_layers=opt.layers,
            dropout=0.3,
            use_cuda=use_cuda,
            enable_embedding_training=bool(opt.enable_embedding_training),
            discriminator_hidden_size=8)

        load_embeddings(model,
                        src_embeddings_filename=opt.src_embeddings,
                        tgt_embeddings_filename=opt.tgt_embeddings,
                        vocabulary=vocabulary)
        model = model.cuda() if use_cuda else model

        trainer = Trainer(opt.src_lang, opt.tgt_lang,
                          use_cuda=use_cuda,
                          discriminator_lr=opt.discriminator_lr,
                          main_lr=opt.learning_rate,
                          main_betas=(opt.adam_beta1, 0.999), )
        pair_file_names = [(train_src_bi, train_tgt_bi), ]
        trainer.train_supervised(model, pair_file_names, vocabulary, batch_size=batch_size, max_len=max_length,
                                 save_file=save_file, big_epochs=supervised_epochs, print_every=print_every,
                                 save_every=save_every, max_batch_count=n_supervised_batches)
        zero_model = Translator(model, vocabulary, use_cuda)
    else:
        raise ValueError('Zero Model was not initialized')
    return zero_model


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    use_cuda = torch.cuda.is_available()
    print("Use CUDA: ", use_cuda)
    trainer = Trainer(opt.src_lang, opt.tgt_lang,
                      use_cuda=use_cuda,
                      discriminator_lr=opt.discriminator_lr,
                      main_lr=opt.learning_rate,
                      main_betas=(opt.adam_beta1, 0.999),)

    _, _, vocabulary = collect_vocabularies(
            src_vocabulary_path=opt.src_vocabulary,
            tgt_vocabulary_path=opt.tgt_vocabulary,
            all_vocabulary_path=opt.all_vocabulary,
            src_file_names=(opt.train_src_mono, ),
            tgt_file_names=(opt.train_tgt_mono, ),
            src_max_words=opt.src_vocab_size,
            tgt_max_words=opt.tgt_vocab_size,
            reset=bool(opt.reset_vocabularies))

    model, discriminator = build_model(
        max_length=50,
        output_size=vocabulary.size(),
        rnn_size=opt.rnn_size,
        encoder_n_layers=opt.layers,
        decoder_n_layers=opt.layers,
        dropout=0.3,
        use_cuda=use_cuda,
        enable_embedding_training=bool(opt.enable_embedding_training),
        discriminator_hidden_size=opt.discriminator_hidden_size)

    load_embeddings(model,
                    src_embeddings_filename=opt.src_embeddings,
                    tgt_embeddings_filename=opt.tgt_embeddings,
                    vocabulary=vocabulary)
    model = model.cuda() if use_cuda else model
    discriminator = discriminator.cuda() if use_cuda else discriminator

    zero_model = init_zero_model(vocabulary, use_cuda,
                                 src_to_tgt_dict=opt.src_to_tgt_dict,
                                 tgt_to_src_dict=opt.tgt_to_src_dict,
                                 bootstrapped_model_path=opt.bootstrapped_model,
                                 train_src_bi=opt.train_src_bi,
                                 train_tgt_bi=opt.train_tgt_bi,
                                 supervised_epochs=opt.supervised_epochs,
                                 batch_size=opt.batch_size,
                                 n_supervised_batches=opt.n_supervised_batches,
                                 print_every=opt.print_every,
                                 save_every=opt.save_every,
                                 save_file=opt.save_model)
    trainer.current_translation_model = zero_model

    if opt.load_from:
        model, discriminator, main_optimizer, discriminator_optimizer = load_model(opt.load_from, use_cuda)
        trainer.main_optimizer = main_optimizer
        trainer.discriminator_optimizer = discriminator_optimizer

    trainer.train(model, discriminator,
                  src_file_names=[opt.train_src_mono, ],
                  tgt_file_names=[opt.train_tgt_mono, ],
                  unsupervised_big_epochs=opt.unsupervised_epochs,
                  batch_size=opt.batch_size,
                  print_every=opt.print_every,
                  save_every=opt.save_every,
                  save_file=opt.save_model,
                  n_unsupervised_batches=opt.n_unsupervised_batches,
                  enable_unsupervised_backtranslation=opt.enable_unsupervised_backtranslation)


if __name__ == "__main__":
    main()
