import torch

from src.models import Seq2Seq, Discriminator, build_model
from src.trainer import init_optimizers


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
                                       use_cuda=use_cuda)
    model.load_state_dict(state_dict['model'])
    discriminator.load_state_dict(state_dict['discriminator'])
    model = model.cuda() if use_cuda else model
    discriminator = discriminator.cuda() if use_cuda else discriminator

    main_optimizer, discriminator_optimizer = init_optimizers(model, discriminator)
    main_optimizer.load_state_dict(state_dict['main_optimizer'])
    discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])

    return model, discriminator, main_optimizer, discriminator_optimizer
