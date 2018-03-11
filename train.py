import argparse
import copy
import sys
import logging

import torch

from src.trainer import Trainer
from src.translator import Translator
from src.models import build_model, load_embeddings, print_summary
from src.word_by_word import WordByWordModel
from utils.vocabulary import collect_vocabularies
from src.serialize import load_model


def train_opts(parser):
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
    group.add_argument('-max_length', type=int, default=50,
                       help="Sentence max length")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')
    group.add_argument('-usv_embedding_training', type=int, default=1,
                       help='Enable embedding training in unsupervised model.')
    group.add_argument('-sv_embedding_training', type=int, default=0,
                       help='Enable embedding training in supervised model.')

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
    group.add_argument('-dropout', type=float, default=0.3,
                       help='Dropout rate')
    group.add_argument('-discriminator_hidden_size', type=int, default=1024,
                       help='Size of discriminator hidden layers')
    group.add_argument('-attention', type=int, default=1,
                       help='Enable attention')
    group.add_argument('-bidirectional', type=int, default=1,
                       help='Enable bidirectional encoder')

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
    group.add_argument('-usv_load_from', default=None, type=str,
                       help="Load unsupervised model from file")
    group.add_argument('-sv_load_from', default=None, type=str,
                       help="Load supervised model from file")
    # Logging
    group = parser.add_argument_group('Logging')
    group.add_argument('-print_every', type=int, default=1000,
                       help='Count of minibatches to print')
    group.add_argument('-log_file', type=str, default="log.txt",
                       help='Log file for debug messages')

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-sv_num_words_in_batch', type=int, default=500,
                       help='Batch size in words for supervised training')
    group.add_argument('-usv_num_words_in_batch', type=int, default=250,
                       help='Batch size in words for unsupervised training')
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
    group.add_argument('-teacher_forcing', type=int, default=1,
                       help='Enable teacher forcing')

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-sv_learning_rate', type=float, default=0.003,
                       help="""Supervised training learning rate.""")
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


def init_zero_supervised(vocabulary, save_file, use_cuda):
    model, discriminator = build_model(
        max_length=opt.max_length,
        output_size=vocabulary.size(),
        rnn_size=opt.rnn_size,
        encoder_n_layers=opt.layers,
        decoder_n_layers=opt.layers,
        dropout=opt.dropout,
        use_cuda=use_cuda,
        enable_embedding_training=bool(opt.sv_embedding_training),
        discriminator_hidden_size=opt.discriminator_hidden_size,
        bidirectional=bool(opt.bidirectional),
        use_attention=bool(opt.attention)
    )
    if opt.src_embeddings is not None:
        load_embeddings(model,
                        src_embeddings_filename=opt.src_embeddings,
                        tgt_embeddings_filename=opt.tgt_embeddings,
                        vocabulary=vocabulary)
    model = model.cuda() if use_cuda else model
    discriminator = discriminator.cuda() if use_cuda else discriminator
    print_summary(model)
    

    trainer = Trainer(vocabulary,
                      max_length=opt.max_length,
                      use_cuda=use_cuda,
                      discriminator_lr=opt.discriminator_lr,
                      main_lr=opt.sv_learning_rate,
                      main_betas=(opt.adam_beta1, 0.999), )
    
    if opt.sv_load_from:
        model, discriminator, main_optimizer, discriminator_optimizer = load_model(opt.sv_load_from, use_cuda)
        trainer.main_optimizer = main_optimizer
        trainer.discriminator_optimizer = discriminator_optimizer
    
    pair_file_names = [(opt.train_src_bi, opt.train_tgt_bi), ]
    trainer.train_supervised(model, discriminator, pair_file_names, vocabulary, num_words_in_batch=opt.sv_num_words_in_batch,
                             max_length=opt.max_length,save_file=save_file, big_epochs=opt.supervised_epochs, 
                             print_every=opt.print_every, save_every=opt.save_every, max_batch_count=opt.n_supervised_batches)
    for param in model.parameters():
        param.requires_grad = False
    return Translator(model, vocabulary, use_cuda)


def main():
    logging.basicConfig(level=logging.DEBUG)
    
    logger = logging.getLogger("unmt")
    logger.propagate = False
    fh = logging.FileHandler(opt.log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    use_cuda = torch.cuda.is_available()
    logger.info("Use CUDA: " + str(use_cuda))
  
    _, _, vocabulary = collect_vocabularies(
            src_vocabulary_path=opt.src_vocabulary,
            tgt_vocabulary_path=opt.tgt_vocabulary,
            all_vocabulary_path=opt.all_vocabulary,
            src_file_names=(opt.train_src_mono, ),
            tgt_file_names=(opt.train_tgt_mono, ),
            src_max_words=opt.src_vocab_size,
            tgt_max_words=opt.tgt_vocab_size,
            reset=bool(opt.reset_vocabularies))

    if opt.src_to_tgt_dict is not None and opt.tgt_to_src_dict is not None:
        zero_model = WordByWordModel(opt.src_to_tgt_dict, opt.tgt_to_src_dict, vocabulary, opt.max_length)
    elif opt.bootstrapped_model is not None:
        model, discriminator, _, _ = load_model(opt.bootstrapped_model, use_cuda)
        for param in model.parameters():
            param.requires_grad = False
        zero_model = Translator(model, vocabulary, use_cuda)
    elif opt.train_src_bi is not None and opt.train_tgt_bi is not None:
        zero_model = init_zero_supervised(vocabulary, opt.save_model, use_cuda)
    else:
        assert False, "Zero model was not initialized"
    
    trainer = Trainer(vocabulary,
                      max_length=opt.max_length,
                      use_cuda=use_cuda,
                      discriminator_lr=opt.discriminator_lr,
                      main_lr=opt.learning_rate,
                      main_betas=(opt.adam_beta1, 0.999),)
    trainer.current_translation_model = zero_model

    model, discriminator = build_model(
        max_length=opt.max_length,
        output_size=vocabulary.size(),
        rnn_size=opt.rnn_size,
        encoder_n_layers=opt.layers,
        decoder_n_layers=opt.layers,
        dropout=opt.dropout,
        use_cuda=use_cuda,
        enable_embedding_training=bool(opt.usv_embedding_training),
        discriminator_hidden_size=opt.discriminator_hidden_size,
        bidirectional=bool(opt.bidirectional),
        use_attention=bool(opt.attention)
    )
    if opt.src_embeddings is not None:
        load_embeddings(model,
                        src_embeddings_filename=opt.src_embeddings,
                        tgt_embeddings_filename=opt.tgt_embeddings,
                        vocabulary=vocabulary)
    model = model.cuda() if use_cuda else model
    print_summary(model)
    print_summary(discriminator)
    discriminator = discriminator.cuda() if use_cuda else discriminator

    if opt.usv_load_from:
        model, discriminator, main_optimizer, discriminator_optimizer = load_model(opt.usv_load_from, use_cuda)
        trainer.main_optimizer = main_optimizer
        trainer.discriminator_optimizer = discriminator_optimizer

    trainer.train(model, discriminator,
                  src_file_names=[opt.train_src_mono, ],
                  tgt_file_names=[opt.train_tgt_mono, ],
                  unsupervised_big_epochs=opt.unsupervised_epochs,
                  num_words_in_batch=opt.usv_num_words_in_batch,
                  print_every=opt.print_every,
                  save_every=opt.save_every,
                  save_file=opt.save_model,
                  n_unsupervised_batches=opt.n_unsupervised_batches,
                  enable_unsupervised_backtranslation=opt.enable_unsupervised_backtranslation,
                  teacher_forcing=bool(opt.teacher_forcing),
                  max_length=opt.max_length)


if __name__ == "__main__":
    main()
