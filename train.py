import argparse

import time
import torch
from src.trainer import Trainer
from src.translator import Translator


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

    # Init options
    # group = parser.add_argument_group('Initialization')
    # group.add_argument('-start_epoch', type=int, default=1,
    #                    help='The epoch from which to start')
    # group.add_argument('-train_from', default='', type=str,
    #                    help="""If training from a checkpoint then this is the
    #                    path to the pretrained model's state_dict.""")

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
    group.add_argument('-discr-learning_rate', type=float, default=0.0005,
                       help="""Discriminator learning rate""")


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
train_opts(parser)
opt = parser.parse_args()


def zero_model(state, use_cuda, print_every=1000, save_every=1000, save_file: str="model"):
    if opt.src_to_tgt_dict is not None and opt.tgt_to_src_dict is not None:
        state.current_translation_model = state.build_word_by_word_model(src_to_tgt_dict_filename=opt.src_to_tgt_dict,                                                                         tgt_to_src_dict_filename=opt.tgt_to_src_dict)

    elif opt.bootstrapped_model is not None:
        state.load(opt.bootstrapped_model)
        state.model = state.model.cuda() if use_cuda else state.model
        state.current_translation_model = state.model
        for param in state.current_translation_model.parameters():
            param.requires_grad = False
        state.load(opt.bootstrapped_model)
        state.model = state.model.cuda() if use_cuda else state.model

    elif opt.train_src_bi is not None and opt.train_tgt_bi is not None:
        batch_size = opt.batch_size
        n_supervised_batches = opt.n_supervised_batches
        pair_filenames = [(opt.train_src_bi, opt.train_tgt_bi), ]
        supervised_big_epochs = opt.supervised_epochs

        parallel_forward_batches = state.get_bilingual_batches(pair_filenames, lang="src",
                                                               batch_size=batch_size, n=n_supervised_batches)
        reverted_pairs = [(pair[1], pair[0]) for pair in pair_filenames]
        reverted_batches = state.get_bilingual_batches(reverted_pairs, lang="tgt",
                                                       batch_size=batch_size, n=n_supervised_batches)
        count_supervised_batches = len(parallel_forward_batches)

        src_batches = state.get_one_lang_batches([opt.train_src_bi, ], lang="src",
                                                 batch_size=batch_size, n=n_supervised_batches)
        tgt_batches = state.get_one_lang_batches([opt.train_tgt_bi, ], lang="tgt",
                                                 batch_size=batch_size, n=n_supervised_batches)

        print("Src batch:", src_batches[0])
        print("Tgt batch:", tgt_batches[0])

        for big_epoch in range(supervised_big_epochs):
            timer = time.time()
            print_loss_total = 0
            for epoch, batch in enumerate(parallel_forward_batches):
                state.model.train()
                loss = state.train_bilingual_batch(batch, reverted_batches[epoch])

                print_loss_total += loss
                if epoch % save_every == 0 and epoch != 0:
                    state.save(save_file + "_supervised.pt")
                if epoch % print_every == 0 and epoch != 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    print('%s big epoch, %s/%s epoch, %s sec, %.4f main loss' %
                          (big_epoch, epoch, count_supervised_batches, diff, print_loss_avg))
            state.save(save_file + "_supervised.pt")
        state.current_translation_model = state.model

    raise ValueError('Zero Model was not initialized')


def main():
    use_cuda = torch.cuda.is_available()
    print("Use CUDA: ", use_cuda)
    state = Trainer(opt.src_lang, opt.tgt_lang, use_cuda=use_cuda)
    state.init_model(src_filenames=[opt.train_src_mono, ],
                     tgt_filenames=[opt.train_tgt_mono, ],
                     src_embeddings_filename=opt.src_embeddings,
                     tgt_embeddings_filename=opt.tgt_embeddings,
                     src_max_words=opt.src_vocab_size,
                     tgt_max_words=opt.tgt_vocab_size,
                     hidden_size=opt.rnn_size,
                     n_layers=opt.layers,
                     discriminator_lr=opt.discr_learning_rate,
                     main_lr=opt.learning_rate,
                     main_betas=(opt.adam_beta1, 0.999),
                     discriminator_hidden_size=opt.discriminator_hidden_size,
                     src_vocabulary_path=opt.src_vocabulary,
                     tgt_vocabulary_path=opt.tgt_vocabulary,
                     all_vocabulary_path=opt.all_vocabulary,
                     enable_embedding_training=opt.enable_embedding_training)

    # Init Zero Model
    zero_model(state, use_cuda)

    print(Translator.translate(state.model, "you can prepare your meals here .", "src", "tgt",
                               state.all_vocabulary, state.use_cuda))

    state.train([opt.train_src_mono, ], [opt.train_tgt_mono, ],
                unsupervised_big_epochs=opt.unsupervised_epochs,
                batch_size=opt.batch_size,
                print_every=opt.print_every,
                save_every=opt.save_every,
                save_file=opt.save_model,
                n_unsupervised_batches=opt.n_unsupervised_batches)


if __name__ == "__main__":
    main()
