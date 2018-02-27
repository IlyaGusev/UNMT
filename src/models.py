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

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)

        packed = pack(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, None)
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


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sm = nn.Softmax(dim=1)

        self.out = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, decoder_rnn_output, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        decoder_rnn_output = decoder_rnn_output.transpose(0, 1)
        energy = self.attn(encoder_outputs).view(batch_size, self.hidden_size, max_len)
        attn_energies = decoder_rnn_output.bmm(energy).transpose(0, 1).squeeze(0)  # S = B x L

        attn_weights = self.sm(attn_energies).unsqueeze(1)  # S = B x 1 x L
        encoder_context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)  # S = 1 x B x N

        concat_context = torch.cat([encoder_context, decoder_rnn_output.transpose(0, 1)], 2)
        context = self.tanh(self.out(concat_context))

        return context, attn_weights.squeeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, n_layers=3, dropout=0.3, max_length=50, use_cuda=False):
        super(AttnDecoderRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attn = Attn(hidden_size)
        self.rnn = nn.LSTM(hidden_size + embedding_dim, hidden_size, n_layers, dropout=dropout)
        self.generator = Generator(hidden_size, output_size)

    def init_state(self, encoder_outputs, sos_index):
        max_encoder_length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        initial_input = Variable(torch.zeros((batch_size,)).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input

        attn_weights = Variable(torch.zeros(batch_size, max_encoder_length)).unsqueeze(1)  # B x 1 x L
        attn_weights = torch.add(attn_weights, 1.0 / max_encoder_length)
        attn_weights = attn_weights.cuda() if self.use_cuda else attn_weights

        initial_context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)  # 1 x B x N
        initial_context = initial_context.cuda() if self.use_cuda else initial_context
        initial_context = initial_context.detach()

        return initial_input, initial_context

    def step(self, batch_input, hidden, context, encoder_output):
        # batch_input: B
        # hidden: n_layers x B x N
        # encoder_output: 1 x B x N
        # output: 1 x B x N
        # embedded:  1 x B x E
        embedded = self.embedding(batch_input).unsqueeze(0)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.rnn(rnn_input, hidden)
        output, attn_weights = self.attn.forward(output, encoder_output)
        return output, hidden, attn_weights

    def forward(self, current_input, context, hidden, length, encoder_output):
        batch_size = encoder_output.size(1)
        padded_encoder_output = Variable(torch.zeros((length, batch_size, self.hidden_size)).type(torch.FloatTensor))
        padded_encoder_output = padded_encoder_output.cuda() if self.use_cuda else padded_encoder_output
        for i in range(encoder_output.size(0)):
            padded_encoder_output[i] = encoder_output[i]

        outputs = Variable(torch.zeros(length, current_input.size(0), self.output_size))
        outputs = outputs.cuda() if self.use_cuda else outputs

        for t in range(length):
            output, hidden, _ = self.step(current_input, hidden, context, encoder_output)
            context = output
            scores = self.generator.forward(output.squeeze(0))
            top_indices = scores.topk(1, dim=1)[1].view(-1)
            current_input = top_indices
            outputs[t] = scores
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
        for i in range(self.n_layers):
            output = self.layers[i](output)
            output = self.activation(output)
        return self.sigmoid(self.out(output))


class Seq2Seq(nn.Module):
    def __init__(self, embedding_dim, rnn_size, output_size, encoder_n_layers, decoder_n_layers, dropout,
                 max_length, use_cuda, enable_embedding_training, bidirectional):
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

        self.encoder = EncoderRNN(self.all_size, embedding_dim, rnn_size, dropout=dropout,
                                  n_layers=encoder_n_layers, bidirectional=bidirectional)
        self.decoder = AttnDecoderRNN(embedding_dim, rnn_size, self.output_size, dropout=dropout,
                                      max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda)

        self.encoder.embedding.weight.requires_grad = enable_embedding_training
        self.decoder.embedding.weight.requires_grad = enable_embedding_training

    def load_embeddings(self, src_embeddings, tgt_embeddings, vocabulary: Vocabulary):
        aligned_embeddings = torch.div(torch.randn(vocabulary.size(), 300), 10)
        for i in range(len(vocabulary.index2word)):
            word = vocabulary.get_word(i)
            language = vocabulary.get_language(i)
            if language == "src" and word in src_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word])
            if language == "tgt" and word in tgt_embeddings.wv:
                aligned_embeddings[i] = torch.FloatTensor(tgt_embeddings.wv[word])

        enable_training = self.encoder.embedding.weight.requires_grad
        self.encoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)
        self.decoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)

    def forward(self, variable, lengths, sos_index):
        encoder_output, encoder_hidden = self.encoder.forward(variable, lengths)
        current_input, context = self.decoder.init_state(encoder_output, sos_index)
        decoder_output, _ = self.decoder.forward(current_input, context, encoder_hidden, max(lengths), encoder_output)

        return encoder_output, decoder_output


def build_model(*, rnn_size, output_size, encoder_n_layers, decoder_n_layers, discriminator_hidden_size, dropout,
                max_length, use_cuda, enable_embedding_training):
    logging.info("Building model...")
    model = Seq2Seq(embedding_dim=300,
                    rnn_size=rnn_size,
                    output_size=output_size,
                    use_cuda=use_cuda,
                    encoder_n_layers=encoder_n_layers,
                    decoder_n_layers=decoder_n_layers,
                    enable_embedding_training=enable_embedding_training,
                    max_length=max_length,
                    dropout=dropout)
    discriminator = Discriminator(max_length=max_length,
                                  encoder_hidden_size=rnn_size,
                                  hidden_size=discriminator_hidden_size,
                                  n_layers=3)
    return model, discriminator


def load_embeddings(model, src_embeddings_filename, tgt_embeddings_filename, vocabulary):
    logging.info("Loading embeddings...")
    src_word_vectors = KeyedVectors.load_word2vec_format(src_embeddings_filename, binary=False)
    tgt_word_vectors = KeyedVectors.load_word2vec_format(tgt_embeddings_filename, binary=False)
    model.load_embeddings(src_word_vectors, tgt_word_vectors, vocabulary)


def print_summary(model):
    logging.info(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Params: ", params)
