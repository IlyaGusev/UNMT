import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors

from src.batch_transformer import BatchTransformer
from src.models import EncoderRNN
from src.unmt import UNMT
from src.word_by_word import WordByWordModel
from utils.batch import OneLangBatch, OneLangBatchGenerator, indices_from_sentence, \
    BilingualBatch, BilingualBatchGenerator
from utils.tqdm import tqdm_open
from utils.vocabulary import Vocabulary
from src.translator import Translator
from src.loss import DiscriminatorLossCompute, MainLossCompute


class Trainer:
    def __init__(self, src_lang: str, tgt_lang: str, max_length: int=50, use_cuda: bool=True):
        self.model = None  # type: UNMT
        self.current_translation_model = None

        self.src_lang = src_lang  # type: str
        self.tgt_lang = tgt_lang  # type: str
        self.max_length = max_length  # type: int
        self.use_cuda = use_cuda  # type: bool

        self.src_vocabulary = None  # type: Vocabulary
        self.tgt_vocabulary = None  # type: Vocabulary
        self.all_vocabulary = None  # type: Vocabulary

        self.criterion = None
        self.discriminator_optimizer = None
        self.main_optimizer = None

        self.adv_ones_variable = None
        self.adv_zeros_variable = None

    def collect_vocabularies(self, src_vocabulary_path: str, tgt_vocabulary_path: str, all_vocabulary_path: str,
                             src_filenames=None, tgt_filenames=None, src_max_words=80000, tgt_max_words=100000,
                             reset=True):
        print("Collecting vocabularies...")
        self.src_vocabulary = Vocabulary(language=self.src_lang, path=src_vocabulary_path)
        self.tgt_vocabulary = Vocabulary(language=self.tgt_lang, path=tgt_vocabulary_path)
        self.all_vocabulary = Vocabulary(language="all", path=all_vocabulary_path)
        if src_filenames is not None and reset:
            self.src_vocabulary.reset()
            self.tgt_vocabulary.reset()
            self.all_vocabulary.reset()

        if self.src_vocabulary.is_empty():
            for filename in src_filenames:
                self.src_vocabulary = self.add_filename_to_vocabulary(filename, self.src_vocabulary)
            self.src_vocabulary.shrink(src_max_words)
            self.src_vocabulary.save(src_vocabulary_path)
        assert self.src_vocabulary.size() > 4

        if self.tgt_vocabulary.is_empty():
            for filename in tgt_filenames:
                self.tgt_vocabulary = self.add_filename_to_vocabulary(filename, self.tgt_vocabulary)
            self.tgt_vocabulary.shrink(tgt_max_words)
            self.tgt_vocabulary.save(tgt_vocabulary_path)
        assert self.tgt_vocabulary.size() > 4

        if self.all_vocabulary.is_empty():
            self.all_vocabulary = Vocabulary.merge(self.src_vocabulary, self.tgt_vocabulary, all_vocabulary_path)
            self.all_vocabulary.save(all_vocabulary_path)
        assert self.all_vocabulary.size() == self.src_vocabulary.size() + self.tgt_vocabulary.size() - 1

    def init_criterions(self):
        weight = torch.ones(self.all_vocabulary.size())
        weight[self.all_vocabulary.get_pad()] = 0
        weight = weight.cuda() if self.use_cuda else weight
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def build_model(self, hidden_size, encoder_n_layers, decoder_n_layers,
                    discriminator_hidden_size, embeddings_freeze=True):
        print("Building model...")
        self.model = UNMT(300, self.all_vocabulary, hidden_size,
                          discriminator_hidden_size=discriminator_hidden_size, use_cuda=self.use_cuda,
                          encoder_n_layers=encoder_n_layers, decoder_n_layers=decoder_n_layers,
                          embeddings_freeze=embeddings_freeze)

    def load_embeddings(self, src_embeddings_filename, tgt_embeddings_filename, enable_training=False):
        print("Loading embeddings...")
        src_word_vectors = KeyedVectors.load_word2vec_format(src_embeddings_filename, binary=False)
        tgt_word_vectors = KeyedVectors.load_word2vec_format(tgt_embeddings_filename, binary=False)
        self.model.load_embeddings(src_word_vectors, tgt_word_vectors, enable_training=enable_training)
        assert self.model.encoder.embedding.weight.size(0) == self.all_vocabulary.size()

    def init_optimizers(self, discriminator_lr=0.0005, main_lr=0.0003, main_betas=(0.5, 0.999)):
        print("Initializing optimizers...")
        self.discriminator_optimizer = optim.RMSprop(self.model.discriminator.parameters(), lr=discriminator_lr)
        self.main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=main_lr, betas=main_betas)

    def print_summary(self):
        print(self.model)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Params: ", params)

    def build_word_by_word_model(self, src_to_tgt_dict_filename, tgt_to_src_dict_filename):
        return WordByWordModel(src_to_tgt_dict_filename, tgt_to_src_dict_filename, self.all_vocabulary)

    def init_model(self, src_filenames=None, tgt_filenames=None,
                   src_embeddings_filename=None, tgt_embeddings_filename=None,
                   src_max_words=80000, tgt_max_words=100000, hidden_size=200, n_layers=3, discriminator_lr=0.0005,
                   main_lr=0.0003, main_betas=(0.5, 0.999), discriminator_hidden_size=512,
                   src_vocabulary_path: str="src.pickle", tgt_vocabulary_path: str="tgt.pickle",
                   all_vocabulary_path: str="all.pickle", enable_embedding_training=False,
                   reset_vocabularies=True):

        self.collect_vocabularies(src_vocabulary_path=src_vocabulary_path,
                                  tgt_vocabulary_path=tgt_vocabulary_path,
                                  all_vocabulary_path=all_vocabulary_path,
                                  src_filenames=src_filenames,
                                  tgt_filenames=tgt_filenames,
                                  src_max_words=src_max_words,
                                  tgt_max_words=tgt_max_words,
                                  reset=reset_vocabularies)
        self.init_criterions()
        self.build_model(hidden_size=hidden_size,
                         encoder_n_layers=n_layers,
                         decoder_n_layers=n_layers,
                         discriminator_hidden_size=discriminator_hidden_size,
                         embeddings_freeze=False)

        if src_embeddings_filename is not None:
            self.load_embeddings(src_embeddings_filename, tgt_embeddings_filename,
                                 enable_training=enable_embedding_training)

        self.model = self.model.cuda() if self.use_cuda else self.model

        self.init_optimizers(discriminator_lr=discriminator_lr,
                             main_lr=main_lr,
                             main_betas=main_betas)

        self.print_summary()

    def train(self, src_filenames, tgt_filenames,
              unsupervised_big_epochs: int, print_every=1000, save_every=1000,
              batch_size: int=32, n_unsupervised_batches: int=None, save_file: str="model",
              enable_unsupervised_backtranslation=False):
        self.adv_ones_variable = Variable(torch.add(torch.ones((batch_size,)), -0.1), requires_grad=False)
        self.adv_ones_variable = self.adv_ones_variable.cuda() if self.use_cuda else self.adv_ones_variable
        self.adv_zeros_variable = Variable(torch.add(torch.zeros((batch_size,)), 0.1), requires_grad=False)
        self.adv_zeros_variable = self.adv_zeros_variable.cuda() if self.use_cuda else self.adv_zeros_variable

        for big_epoch in range(unsupervised_big_epochs):
            src_batches = self.get_one_lang_batches(src_filenames, "src", batch_size, n=n_unsupervised_batches)
            tgt_batches = self.get_one_lang_batches(tgt_filenames, "tgt", batch_size, n=n_unsupervised_batches)
            print("Src batch:", next(src_batches))
            print("Tgt batch:", next(tgt_batches))
            timer = time.time()
            print_main_loss_total = 0
            print_discriminator_loss_total = 0
            epoch = 0
            for src_batch, tgt_batch in zip(src_batches, tgt_batches):
                self.model.train()
                discriminator_loss, losses = self.train_batch(src_batch, tgt_batch)
                main_loss = sum(losses)

                print_main_loss_total += main_loss
                print_discriminator_loss_total += discriminator_loss
                if epoch % save_every == 0 and epoch != 0:
                    self.save(save_file + ".pt")
                if epoch % print_every == 0 and epoch != 0:
                    print_main_loss_avg = print_main_loss_total / print_every
                    print_discriminator_loss_avg = print_discriminator_loss_total / print_every
                    print_main_loss_total = 0
                    print_discriminator_loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    # print(Translator.translate(self.model, "you can prepare your meals here .", "src", "src",
                    #                            self.all_vocabulary, self.use_cuda))
                    # print(Translator.translate(self.model, "you can prepare your meals here .", "src", "tgt",
                    #                            self.all_vocabulary, self.use_cuda))
                    print('%s big epoch, %s epoch, %s sec, %.4f main loss, '
                          '%.4f discriminator loss, current losses: %s' %
                          (big_epoch, epoch, diff, print_main_loss_avg, print_discriminator_loss_avg, losses))
                epoch += 1
            self.save(save_file+".pt")
            if enable_unsupervised_backtranslation:
                self.current_translation_model = self.model

    def get_one_lang_batches(self, filenames, lang, batch_size: int=32, n=None):
        batch_generator = OneLangBatchGenerator(filenames, batch_size, self.max_length, self.all_vocabulary, lang)
        i = 0
        for batch in batch_generator:
            yield batch
            if n is not None and i == n:
                break
            i += 1

    def get_bilingual_batches(self, filenames, lang, batch_size: int=32, n=None):
        batch_generator = BilingualBatchGenerator(filenames, batch_size, self.max_length,
                                                  self.all_vocabulary, lang, self.use_cuda)
        i = 0
        for batch in batch_generator:
            yield batch
            if n is not None and i == n:
                break
            i += 1

    def train_batch(self, src_batch: OneLangBatch, tgt_batch: OneLangBatch):
        assert self.model.training
        src_batch = src_batch.cuda() if self.use_cuda else src_batch
        tgt_batch = tgt_batch.cuda() if self.use_cuda else tgt_batch

        discriminator_loss = self.discriminator_step(src_batch, tgt_batch)

        # Main step
        input_batches = dict()
        gtruth_batches = dict()
        sos_indices = dict()

        input_batches["auto-src"], gtruth_batches["auto-src"] = BatchTransformer.noise(src_batch)
        input_batches["auto-tgt"], gtruth_batches["auto-tgt"] = BatchTransformer.noise(tgt_batch)

        translation_func = self.current_translation_model.translate_to_tgt
        input_batches["cd-src"], gtruth_batches["cd-src"] = \
            BatchTransformer.translate_with_noise(src_batch, translation_func)
        translation_func = self.current_translation_model.translate_to_src
        input_batches["cd-tgt"], gtruth_batches["cd-tgt"] = \
            BatchTransformer.translate_with_noise(tgt_batch, translation_func)

        if True:
            print("Src noisy batch: ", input_batches["auto-src"])
            print("Src old new batch: ", gtruth_batches["auto-src"])
            print("Tgt noisy batch: ", input_batches["auto-tgt"])
            print("Tgt old new batch: ", gtruth_batches["auto-tgt"])
            print("Src noisy translated batch: ", input_batches["cd-src"])
            print("Src old new untranslated batch: ", gtruth_batches["cd-src"])
            print("Tgt noisy translated batch: ", input_batches["cd-tgt"])
            print("Tgt old new untranslated batch: ", gtruth_batches["cd-tgt"])

        for key in gtruth_batches:
            gtruth_batches[key] = gtruth_batches[key].cuda() if self.use_cuda else gtruth_batches[key]
        for key in gtruth_batches:
            input_batches[key] = input_batches[key].cuda() if self.use_cuda else input_batches[key]
        for key in gtruth_batches:
            sos_indices[key] = self.all_vocabulary.get_lang_sos(key[-3:])

        adv_targets = dict()
        adv_targets["auto-src"] = self.adv_ones_variable
        adv_targets["auto-tgt"] = self.adv_zeros_variable
        adv_targets["cd-src"] = self.adv_zeros_variable
        adv_targets["cd-tgt"] = self.adv_ones_variable

        self.main_optimizer.zero_grad()

        results = self.model(input_batches, sos_indices, gtruth_batches)

        main_loss_computer = MainLossCompute(self.model.generator, self.all_vocabulary)
        adv_loss_computer = DiscriminatorLossCompute(self.model.discriminator)
        losses = []
        for key, result in results.items():
            main_loss = main_loss_computer.compute_loss(result.decoder_output, gtruth_batches[key])
            adv_loss = adv_loss_computer.compute_loss(result.encoder_output, adv_targets[key])
            losses += [main_loss, adv_loss]
        loss = sum(losses)
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 5)
        self.main_optimizer.step()

        losses = [loss.data[0] for loss in losses]
        return discriminator_loss, losses

    def train_bilingual_batch(self, batch: BilingualBatch, reverted_batch: BilingualBatch):
        assert self.model.training
        batch = batch.cuda() if self.use_cuda else batch
        reverted_batch = reverted_batch.cuda() if self.use_cuda else reverted_batch

        input_batches = dict()
        gtruth_batches = dict()
        sos_indices = dict()
        input_batches["src-tgt"] = gtruth_batches["tgt-src"] = batch
        input_batches["tgt-src"] = gtruth_batches["src-tgt"] = reverted_batch
        for key in gtruth_batches:
            sos_indices[key] = self.all_vocabulary.get_lang_sos(key[-3:])
        self.main_optimizer.zero_grad()

        results = self.model(input_batches, sos_indices, gtruth_batches)

        main_loss_computer = MainLossCompute(self.model.generator, self.all_vocabulary)
        losses = []
        for key, result in results.items():
            main_loss = main_loss_computer.compute_loss(result.decoder_output, gtruth_batches[key])
            losses += [main_loss]
        loss = sum(losses)
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 5)
        self.main_optimizer.step()

        return loss.data[0]

    def discriminator_step(self, src_batch, tgt_batch):
        self.discriminator_optimizer.zero_grad()
        src_adv_loss = self.get_discriminator_loss(batch=src_batch, encoder=self.model.encoder,
                                                   target_variable=self.adv_zeros_variable)
        tgt_adv_loss = self.get_discriminator_loss(batch=tgt_batch, encoder=self.model.encoder,
                                                   target_variable=self.adv_ones_variable)
        discriminator_loss = src_adv_loss + tgt_adv_loss
        discriminator_loss.backward()
        nn.utils.clip_grad_norm(self.model.discriminator.parameters(), 5)
        self.discriminator_optimizer.step()
        return discriminator_loss.data[0]

    def get_discriminator_loss(self, batch: OneLangBatch, encoder: EncoderRNN, target_variable: Variable):
        adv_criterion = nn.BCELoss()
        encoder_output, _ = encoder(batch.variable, batch.lengths, None)
        log_prob = self.model.discriminator(encoder_output).view(-1)
        return adv_criterion(log_prob, target_variable)

    @staticmethod
    def save_model(model: UNMT, discriminator_optimizer, main_optimizer, filename):
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'encoder_n_layers': model.encoder_n_layers,
            'decoder_n_layers': model.decoder_n_layers,
            'hidden_size': model.hidden_size,
            'discriminator_hidden_size': model.discriminator_hidden_size,
            'embeddings_freeze': model.encoder.embedding.weight.requires_grad,
            'state_dict': state_dict,
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'main_optimizer': main_optimizer.state_dict(),
        }, filename)

    def save(self, model_filename):
        Trainer.save_model(self.model, self.discriminator_optimizer, self.main_optimizer, model_filename)

    def load(self, model_filename):
        state_dict = torch.load(model_filename)
        self.build_model(hidden_size=state_dict['hidden_size'],
                         encoder_n_layers=state_dict['encoder_n_layers'],
                         decoder_n_layers=state_dict['encoder_n_layers'],
                         discriminator_hidden_size=state_dict['discriminator_hidden_size'],
                         embeddings_freeze=state_dict['embeddings_freeze'])

        self.init_optimizers()
        self.model.load_state_dict(state_dict['state_dict'])
        self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
        self.main_optimizer.load_state_dict(state_dict['main_optimizer'])

    @staticmethod
    def add_filename_to_vocabulary(filename: str, vocabulary: Vocabulary):
        with tqdm_open(filename, encoding="utf-8") as r:
            for line in r:
                for word in line.strip().split():
                    vocabulary.add_word(word)
        return vocabulary
