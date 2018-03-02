import time
import logging
from typing import List, Dict
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from src.batch import Batch, BatchGenerator, BilingualBatchGenerator
from src.batch_transformer import BatchTransformer
from src.loss import DiscriminatorLossCompute, MainLossCompute
from src.models import Seq2Seq, Discriminator
from src.translator import Translator
from src.serialize import save_model
from utils.vocabulary import Vocabulary

class Trainer:
    def __init__(self, vocabulary: Vocabulary, max_length: int=50, use_cuda: bool=True,
                 discriminator_lr=0.0005, main_lr=0.0003, main_betas=(0.5, 0.999)):
        self.vocabulary = vocabulary  # type: Vocabulary
        self.max_length = max_length  # type: int
        self.use_cuda = use_cuda  # type: bool

        self.discriminator_lr = discriminator_lr
        self.main_lr = main_lr
        self.main_betas = main_betas

        self.criterion = None
        self.discriminator_optimizer = None
        self.main_optimizer = None

        self.adv_ones_variable = None
        self.adv_zeros_variable = None

        self.current_translation_model = None

    def train(self, model: Seq2Seq, discriminator: Discriminator,
              src_file_names: List[str], tgt_file_names: List[str],
              unsupervised_big_epochs: int, print_every=1000, save_every=1000,
              batch_size: int=32, n_unsupervised_batches: int=None, save_file: str="model",
              enable_unsupervised_backtranslation=False, max_len: int=50):
        if self.main_optimizer is None or self.discriminator_optimizer is None:
            logging.info("Initializing optimizers...")
            self.main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.main_lr, betas=self.main_betas)
            self.discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=self.discriminator_lr)
        self.adv_ones_variable = Variable(torch.add(torch.ones((batch_size,)), -0.1), requires_grad=False)
        self.adv_ones_variable = self.adv_ones_variable.cuda() if self.use_cuda else self.adv_ones_variable
        self.adv_zeros_variable = Variable(torch.add(torch.zeros((batch_size,)), 0.1), requires_grad=False)
        self.adv_zeros_variable = self.adv_zeros_variable.cuda() if self.use_cuda else self.adv_zeros_variable

        for big_epoch in range(unsupervised_big_epochs):
            src_batch_gen = BatchGenerator(src_file_names, batch_size, max_len=max_len,
                                           vocabulary=self.vocabulary, language="src",
                                           max_batch_count=n_unsupervised_batches)
            tgt_batch_gen = BatchGenerator(tgt_file_names, batch_size, max_len=max_len,
                                           vocabulary=self.vocabulary, language="tgt",
                                           max_batch_count=n_unsupervised_batches)
            logging.debug("Src batch:", next(iter(src_batch_gen)))
            logging.debug("Tgt batch:", next(iter(tgt_batch_gen)))
            timer = time.time()
            main_loss_total = 0
            discriminator_loss_total = 0
            epoch = 0
            for src_batch, tgt_batch in zip(src_batch_gen, tgt_batch_gen):
                model.train()
                discriminator_loss, losses = self.train_batch(model, discriminator, src_batch, tgt_batch)

                main_loss = sum(losses)
                main_loss_total += main_loss
                discriminator_loss_total += discriminator_loss
                if epoch % save_every == 0 and epoch != 0:
                    save_model(model, discriminator, self.main_optimizer,
                               self.discriminator_optimizer, save_file + ".pt")
                if epoch % print_every == 0 and epoch != 0:
                    main_loss_avg = main_loss_total / print_every
                    discriminator_loss_avg = discriminator_loss_total / print_every
                    main_loss_total = 0
                    discriminator_loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    translator = Translator(model, self.vocabulary, self.use_cuda)
                    logging.debug(translator.translate("you can prepare your meals here .", "src", "src"))
                    logging.debug(translator.translate("you can prepare your meals here .", "src", "tgt"))
                    logging.info('%s big epoch, %s epoch, %s sec, %.4f main loss, '
                                 '%.4f discriminator loss, current losses: %s' %
                                 (big_epoch, epoch, diff, main_loss_avg, discriminator_loss_avg, losses))
                epoch += 1
            save_model(model, discriminator, self.main_optimizer,
                       self.discriminator_optimizer, save_file + ".pt")
            if enable_unsupervised_backtranslation:
                self.current_translation_model = Translator(model, self.vocabulary, self.use_cuda)
                model = copy.deepcopy(model)

    def train_batch(self, model: Seq2Seq, discriminator: Discriminator, src_batch: Batch, tgt_batch: Batch):
        src_batch = src_batch.cuda() if self.use_cuda else src_batch
        tgt_batch = tgt_batch.cuda() if self.use_cuda else tgt_batch

        input_batches = dict()
        gtruth_batches = dict()
        sos_indices = dict()

        input_batches["auto-src"], gtruth_batches["auto-src"] =\
            Batch.sort_pair(BatchTransformer.noise(src_batch, self.vocabulary.get_pad("src")), src_batch)
        input_batches["auto-tgt"], gtruth_batches["auto-tgt"] = \
            Batch.sort_pair(BatchTransformer.noise(tgt_batch, self.vocabulary.get_pad("tgt")), tgt_batch)

        translation_func = self.current_translation_model.translate_to_tgt
        input_batches["cd-src"], gtruth_batches["cd-src"] = \
            Batch.sort_pair(BatchTransformer.translate(src_batch, self.vocabulary.get_pad("src"),
                                                       self.vocabulary.get_pad("tgt"), translation_func), src_batch)
        translation_func = self.current_translation_model.translate_to_src
        input_batches["cd-tgt"], gtruth_batches["cd-tgt"] = \
            Batch.sort_pair(BatchTransformer.translate(tgt_batch, self.vocabulary.get_pad("tgt"),
                                                       self.vocabulary.get_pad("src"), translation_func), tgt_batch)

        logging.debug("Src noisy batch: " + str(input_batches["auto-src"]))
        logging.debug("Src old new batch: "+ str(gtruth_batches["auto-src"]))
        logging.debug("Tgt noisy batch: "+ str(input_batches["auto-tgt"]))
        logging.debug("Tgt old new batch: "+ str(gtruth_batches["auto-tgt"]))
        logging.debug("Src noisy translated batch: "+ str(input_batches["cd-src"]))
        logging.debug("Src old new untranslated batch: "+ str(gtruth_batches["cd-src"]))
        logging.debug("Tgt noisy translated batch: "+ str(input_batches["cd-tgt"]))
        logging.debug("Tgt old new untranslated batch: "+ str(gtruth_batches["cd-tgt"]))

        for key in gtruth_batches:
            gtruth_batches[key] = gtruth_batches[key].cuda() if self.use_cuda else gtruth_batches[key]
        for key in gtruth_batches:
            input_batches[key] = input_batches[key].cuda() if self.use_cuda else input_batches[key]

        adv_targets = dict()
        adv_targets["auto-src"] = self.adv_zeros_variable
        adv_targets["auto-tgt"] = self.adv_ones_variable
        adv_targets["cd-src"] = self.adv_ones_variable
        adv_targets["cd-tgt"] = self.adv_zeros_variable

        discriminator_loss = self.discriminator_step(model, discriminator, input_batches, adv_targets)

        adv_targets = dict()
        adv_targets["auto-src"] = self.adv_ones_variable
        adv_targets["auto-tgt"] = self.adv_zeros_variable
        adv_targets["cd-src"] = self.adv_zeros_variable
        adv_targets["cd-tgt"] = self.adv_ones_variable

        for key in gtruth_batches:
            sos_indices[key] = self.vocabulary.get_sos(key[-3:])

        main_losses = self.main_step(model, discriminator, input_batches, gtruth_batches, adv_targets, sos_indices)

        return discriminator_loss, main_losses

    def main_step(self, model: Seq2Seq, discriminator: Discriminator,
                  input_batches: Dict[str, Batch], gtruth_batches: Dict[str, Batch],
                  adv_targets: Dict[str, Variable], sos_indices: Dict[str, int]):
        model.train()
        discriminator.eval()
        self.main_optimizer.zero_grad()
        results = dict()
        for key in input_batches:
            input_batch = input_batches[key]
            sos_index = sos_indices[key]
            results[key] = model.forward(input_batch.variable, input_batch.lengths, sos_index)

        main_loss_computer = MainLossCompute(self.vocabulary, self.use_cuda)
        adv_loss_computer = DiscriminatorLossCompute(discriminator)
        losses = []
        for key, result in results.items():
            main_loss = main_loss_computer.compute(result[1], gtruth_batches[key].variable)
            adv_loss = adv_loss_computer.compute(result[0], adv_targets[key])
            losses += [main_loss, adv_loss]
        loss = sum(losses)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        self.main_optimizer.step()
        return [l.data[0] for l in losses]

    def discriminator_step(self, model: Seq2Seq, discriminator: Discriminator,
                           input_batches: Dict[str, Batch], adv_targets: Dict[str, Variable]):
        discriminator.train()
        model.eval()
        self.discriminator_optimizer.zero_grad()
        adv_loss_computer = DiscriminatorLossCompute(discriminator)

        losses = []
        for key in input_batches:
            input_batch = input_batches[key]
            target = adv_targets[key]
            encoder_output = model.encoder(input_batch.variable, input_batch.lengths)
            losses.append(adv_loss_computer.compute(encoder_output, target))

        discriminator_loss = sum(losses)
        discriminator_loss.backward()
        nn.utils.clip_grad_norm(discriminator.parameters(), 5)
        self.discriminator_optimizer.step()
        return discriminator_loss.data[0]

    def train_supervised(self, model, discriminator, pair_file_names, vocabulary: Vocabulary, *, batch_size, big_epochs, max_len,
                         max_batch_count=None, save_every=100, print_every=100, save_file="model"):
        if self.main_optimizer is None:
            logging.info("Initializing optimizers...")
            self.main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                             lr=self.main_lr, betas=self.main_betas)
            self.discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=self.discriminator_lr)
        for big_epoch in range(big_epochs):
            batch_gen = BilingualBatchGenerator(pair_file_names, batch_size, max_len, vocabulary,
                                                languages=["src", "tgt"], max_batch_count=max_batch_count)
            src_batch, tgt_batch = next(iter(batch_gen))
            logging.debug("Src batch: " + str(src_batch))
            logging.debug("Tgt batch: " + str(tgt_batch))
            src_batch, tgt_batch = next(iter(batch_gen))
            logging.debug("Src batch: " + str(src_batch))
            logging.debug("Tgt batch: " + str(tgt_batch))
            timer = time.time()
            loss_total = 0
            epoch = 0
            model.train()
            for src_batch, tgt_batch in batch_gen:
                loss_total += self.train_supervised_batch(model, src_batch, tgt_batch)
                if epoch % save_every == 0 and epoch != 0:
                    save_model(model, discriminator, self.main_optimizer, self.discriminator_optimizer,
                               save_file + "_supervised.pt")
                if epoch % print_every == 0 and epoch != 0:
                    print_loss_avg = loss_total / print_every
                    loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    print('%s big epoch, %s epoch, %s sec, %.4f main loss' %
                          (big_epoch, epoch, diff, print_loss_avg))
                epoch += 1
            save_model(model,discriminator, self.main_optimizer, self.discriminator_optimizer,
                       save_file + "_supervised.pt")

    def train_supervised_batch(self, model: Seq2Seq, src_batch: Batch, tgt_batch: Batch):
        tgt_reverted_batch, src_reverted_batch = Batch.sort_pair(tgt_batch, src_batch)
        src_batch = src_batch.cuda() if self.use_cuda else src_batch
        tgt_batch = tgt_batch.cuda() if self.use_cuda else tgt_batch
        src_reverted_batch = src_reverted_batch.cuda() if self.use_cuda else src_reverted_batch
        tgt_reverted_batch = tgt_reverted_batch.cuda() if self.use_cuda else tgt_reverted_batch

        input_batches = dict()
        gtruth_batches = dict()
        sos_indices = dict()
        input_batches["src-tgt"] = src_batch
        gtruth_batches["src-tgt"] = tgt_batch
        input_batches["tgt-src"] = tgt_reverted_batch
        gtruth_batches["tgt-src"] = src_reverted_batch
        for key in gtruth_batches:
            sos_indices[key] = self.vocabulary.get_sos(key[-3:])
        self.main_optimizer.zero_grad()
        
        losses = []
        main_loss_computer = MainLossCompute(self.vocabulary, self.use_cuda)
        for key in gtruth_batches:
            _, output = model.forward(input_batches[key].variable, input_batches[key].lengths, sos_indices[key])
            losses.append(main_loss_computer.compute(output, gtruth_batches[key].variable))

        loss = sum(losses)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        self.main_optimizer.step()

        return loss.data[0]