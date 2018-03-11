# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Model save/load.

import torch
from torch import optim
import logging

from src.models import Seq2Seq, Discriminator, build_model


def init_optimizers(model: Seq2Seq, discriminator: Discriminator,
                    discriminator_lr=0.0005, main_lr=0.0003, main_betas=(0.5, 0.999)):
    logging.info("Initializing optimizers...")
    main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=main_lr, betas=main_betas)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=discriminator_lr)
    return main_optimizer, discriminator_optimizer


def save_model(model: Seq2Seq, discriminator: Discriminator, main_optimizer, discriminator_optimizer, filename):
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    discriminator_state_dict = discriminator.state_dict()
    for key in discriminator_state_dict.keys():
        discriminator_state_dict[key] = discriminator_state_dict[key].cpu()
    torch.save({
        'model': model_state_dict,
        'encoder_n_layers': model.encoder_n_layers,
        'decoder_n_layers': model.decoder_n_layers,
        'rnn_size': model.rnn_size,
        'dropout': model.dropout,
        'output_size': model.output_size,
        'embedding_dim': model.embedding_dim,
        'bidirectional': model.bidirectional,
        'attention': model.use_attention,
        'max_length': model.max_length,
        'enable_embedding_training': model.enable_embedding_training,

        'discriminator': discriminator_state_dict,
        'discriminator_hidden_size': discriminator.hidden_size,
        'discriminator_n_layers': discriminator.n_layers,

        'main_optimizer': main_optimizer.state_dict(),
        'discriminator_optimizer': discriminator_optimizer.state_dict()
    }, filename)


def load_model(model_filename, use_cuda):
    state_dict = torch.load(model_filename)
    model, discriminator = build_model(rnn_size=state_dict['rnn_size'],
                                       output_size=state_dict['output_size'],
                                       encoder_n_layers=state_dict['encoder_n_layers'],
                                       decoder_n_layers=state_dict['decoder_n_layers'],
                                       dropout=state_dict['dropout'],
                                       discriminator_hidden_size=state_dict['discriminator_hidden_size'],
                                       max_length=state_dict['max_length'],
                                       enable_embedding_training=state_dict['enable_embedding_training'],
                                       use_cuda=use_cuda,
                                       bidirectional=state_dict['bidirectional'],
                                       use_attention=state_dict['attention'])
    model.load_state_dict(state_dict['model'])
    discriminator.load_state_dict(state_dict['discriminator'])
    model = model.cuda() if use_cuda else model
    discriminator = discriminator.cuda() if use_cuda else discriminator

    main_optimizer, discriminator_optimizer = init_optimizers(model, discriminator)
    main_optimizer.load_state_dict(state_dict['main_optimizer'])
    discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])

    return model, discriminator, main_optimizer, discriminator_optimizer
