# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: PyTorch models

import logging
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from utils.vocabulary import Vocabulary

logger = logging.getLogger("unmt")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, n_layers=3, dropout=0.3, bidirectional=True):
        super(EncoderRNN, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)

        packed = pack(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = unpack(outputs)

        if self.bidirectional:
            n = hidden[0].size(0)
            hidden = (torch.cat([hidden[0][0:n:2], hidden[0][1:n:2]], 2),
                      torch.cat([hidden[1][0:n:2], hidden[1][1:n:2]], 2))
        return outputs, hidden


class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        assert inputs.size(1) == self.hidden_size
        return self.sm(self.out(inputs))


class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, max_length, n_layers=3, 
                 dropout=0.3, use_cuda=False, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.max_length = max_length
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        if self.use_attention:
            self.attn = nn.Linear(hidden_size + embedding_dim, self.max_length, bias=False)
            self.attn_sm = nn.Softmax(dim=1)
            self.attn_out = nn.Linear(hidden_size + embedding_dim, embedding_dim, bias=False)
            self.attn_out_relu = nn.ReLU()
        
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout)
        self.generator = Generator(hidden_size, output_size)
        
    def step(self, batch_input, hidden, encoder_output):
        # batch_input: B
        # hidden: (n_layers x B x N, n_layers x B x N)
        # encoder_output: L x B x N
        # output: 1 x B x N
        # embedded:  B x E
        # attn_weights: B x 1 x L
        # context: B x 1 x N
        # rnn_input: B x N
        
        embedded = self.embedding(batch_input)
        
        if self.use_attention:
            attn_weights = self.attn_sm(self.attn(torch.cat((embedded, hidden[0][-1]), 1))).unsqueeze(1)
            max_length = encoder_output.size(0)
            context = torch.bmm(attn_weights[:, :, :max_length], encoder_output.transpose(0, 1))
            rnn_input = torch.cat((embedded, context.squeeze(1)), 1)
            rnn_input = self.attn_out_relu(self.attn_out(rnn_input))
        else:
            rnn_input = embedded
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden)
        return output, hidden

    def init_state(self, batch_size, sos_index):
        initial_input = Variable(torch.zeros((batch_size,)).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, current_input, hidden, length, encoder_output, gtruth=None):
        outputs = Variable(torch.zeros(length, current_input.size(0), self.output_size), requires_grad=False)
        outputs = outputs.cuda() if self.use_cuda else outputs

        for t in range(length):
            output, hidden = self.step(current_input, hidden, encoder_output)
            scores = self.generator.forward(output.squeeze(0))
            outputs[t] = scores
            if gtruth is None:
                top_indices = scores.topk(1, dim=1)[1].view(-1)
                current_input = top_indices
            else:
                current_input = gtruth[t]
        return outputs, hidden


class Discriminator(nn.Module):
    def __init__(self, max_length, encoder_hidden_size, hidden_size, n_layers):
        super(Discriminator, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_length = max_length

        layers = list()
        layers.append(nn.Linear(encoder_hidden_size * max_length, hidden_size))
        layers.append(nn.LeakyReLU())
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output):
        max_length = encoder_output.size(0)
        batch_size = encoder_output.size(1)
        output = encoder_output.transpose(0, 1).contiguous().view(batch_size, max_length * self.encoder_hidden_size)
        output = F.pad(output, (0, (self.max_length - max_length) * self.encoder_hidden_size), "constant", 0)
        # S = batch_size, max_length * encoder_hidden_size
        for i in range(len(self.layers)):
            output = self.layers[i](output)
        return self.sigmoid(self.out(output))


class Seq2Seq(nn.Module):
    def __init__(self, embedding_dim, rnn_size, output_size, encoder_n_layers, decoder_n_layers, dropout,
                 max_length, use_cuda, enable_embedding_training, bidirectional, use_attention=True):
        super(Seq2Seq, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.bidirectional = bidirectional
        self.enable_embedding_training = enable_embedding_training
        self.use_attention = use_attention

        self.encoder = EncoderRNN(self.output_size, embedding_dim, rnn_size, dropout=dropout,
                                  n_layers=encoder_n_layers, bidirectional=bidirectional)
        self.decoder = DecoderRNN(embedding_dim, rnn_size, self.output_size, dropout=dropout,
                                  max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda, 
                                  use_attention=use_attention)

        self.encoder.embedding.weight.requires_grad = enable_embedding_training
        self.decoder.embedding.weight.requires_grad = enable_embedding_training

    def load_embeddings(self, src_embeddings, tgt_embeddings, vocabulary: Vocabulary):
        aligned_embeddings = torch.div(torch.randn(vocabulary.size(), 300), 10)
        found_count = 0
        for i in range(len(vocabulary.index2word)):
            word = vocabulary.get_word(i)
            language = vocabulary.get_language(i)
            if language == "src" and word in src_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word])
                found_count += 1
            elif language == "src" and word.lower() in src_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word.lower()])
                found_count += 1
                
            if language == "tgt" and word in tgt_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(tgt_embeddings.wv[word])
                found_count += 1
            elif language == "tgt" and word.lower() in tgt_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(tgt_embeddings.wv[word.lower()])
                found_count += 1
        logger.info("Embeddings filled: " + str(found_count) + " of " + str(vocabulary.size()))

        enable_training = self.encoder.embedding.weight.requires_grad
        self.encoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)
        self.decoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)

    def forward(self, variable, lengths, sos_index, gtruth=None):
        encoder_output, encoder_hidden = self.encoder.forward(variable, lengths)
        current_input = self.decoder.init_state(variable.size(1), sos_index)
        max_length = self.max_length
        if gtruth is not None:
            max_length = min(self.max_length, gtruth.size(0))
        decoder_output, _ = self.decoder.forward(current_input, encoder_hidden, max_length,
                                                 encoder_output, gtruth)

        return encoder_output, decoder_output


def build_model(*, rnn_size, output_size, encoder_n_layers, decoder_n_layers, discriminator_hidden_size, dropout,
                max_length, use_cuda, enable_embedding_training, use_attention, bidirectional):
    logger.info("Building model...")
    model = Seq2Seq(embedding_dim=300,
                    rnn_size=rnn_size,
                    output_size=output_size,
                    use_cuda=use_cuda,
                    encoder_n_layers=encoder_n_layers,
                    decoder_n_layers=decoder_n_layers,
                    enable_embedding_training=enable_embedding_training,
                    max_length=max_length,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    use_attention=use_attention)
    discriminator = Discriminator(max_length=max_length,
                                  encoder_hidden_size=rnn_size,
                                  hidden_size=discriminator_hidden_size,
                                  n_layers=3)
    return model, discriminator


def load_embeddings(model, src_embeddings_filename, tgt_embeddings_filename, vocabulary):
    logger.info("Loading embeddings...")
    src_word_vectors = KeyedVectors.load_word2vec_format(src_embeddings_filename, binary=False)
    tgt_word_vectors = KeyedVectors.load_word2vec_format(tgt_embeddings_filename, binary=False)
    model.load_embeddings(src_word_vectors, tgt_word_vectors, vocabulary)


def print_summary(model):
    logger.info(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Params: " + str(params))
