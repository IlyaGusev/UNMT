from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.models import EncoderRNN, Generator, AttnDecoderRNN, Discriminator

RunResult = namedtuple("RunResult", "encoder_output decoder_output attn_weights output_variable")


class UNMT(nn.Module):
    def __init__(self, embedding_dim, all_vocabulary, hidden_size,
                 discriminator_hidden_size=1024, encoder_n_layers=3, decoder_n_layers=3, dropout=0.1,
                 max_length=50, use_cuda=True, embeddings_freeze=True):
        super(UNMT, self).__init__()

        self.embedding_dim = embedding_dim
        self.all_vocabulary = all_vocabulary
        self.all_size = all_vocabulary.size()
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.discriminator_hidden_size = discriminator_hidden_size
        self.use_cuda = use_cuda

        self.encoder = EncoderRNN(self.all_size, embedding_dim, hidden_size, dropout=dropout,
                                  n_layers=encoder_n_layers)
        self.decoder = AttnDecoderRNN(embedding_dim, hidden_size, self.all_size, dropout=dropout,
                                      max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda)
        self.generator = Generator(hidden_size, self.all_size)
        self.discriminator = Discriminator(self.max_length, self.hidden_size, hidden_size=discriminator_hidden_size)

        self.encoder.embedding.weight.requires_grad = embeddings_freeze
        self.decoder.embedding.weight.requires_grad = embeddings_freeze

    def load_embeddings(self, src_embeddings, tgt_embeddings):
        aligned_embeddings = torch.div(torch.randn(self.all_vocabulary.size(), 300), 10)
        for i, word in enumerate(self.all_vocabulary.index2word):
            if "src-" == word[:4]:
                word = word[4:]
                if word in src_embeddings.wv:
                    aligned_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word])
            if "tgt-" == word[:4]:
                word = word[4:]
                if word in tgt_embeddings.wv:
                    aligned_embeddings[i] = torch.FloatTensor(tgt_embeddings.wv[word])

        enable_training = self.encoder.embedding.weight.requires_grad
        self.encoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)
        self.decoder.embedding.weight = nn.Parameter(aligned_embeddings, requires_grad=enable_training)

    def forward(self, input_batches, sos_indices, gtruth_batches=None):
        assert (gtruth_batches is not None) == self.training
        results = dict()
        for key in input_batches:
            input_batch = input_batches[key]
            sos_index = sos_indices[key]
            if gtruth_batches is not None:
                gtruth_batch = gtruth_batches[key]
                results[key] = self.encoder_decoder_run(self.encoder, self.decoder, self.generator,
                                                        input_batch.variable, input_batch.lengths,
                                                        sos_index, gtruth_batch.variable, gtruth_batch.lengths)
            else:
                results[key] = self.encoder_decoder_run(self.encoder, self.decoder, self.generator,
                                                        input_batch.variable, input_batch.lengths,
                                                        sos_index)
        return results

    def encoder_decoder_run(self, encoder, decoder, generator, variable, lengths, sos_index,
                            gtruth_variable=None, gtruth_lengths=None):
        assert (gtruth_variable is not None) == self.training
        assert (gtruth_lengths is not None) == self.training

        encoder_output, encoder_hidden = encoder(variable, lengths)
        current_input, context = decoder.init_state(encoder_output, sos_index)
        decoder_hidden = encoder_hidden

        max_encoder_length = encoder_output.size(0)
        batch_size = encoder_output.size(1)

        decoder_output = Variable(torch.zeros(max_encoder_length, batch_size, self.hidden_size), requires_grad=False)
        decoder_output = decoder_output.cuda() if self.use_cuda else decoder_output
        attn_weights = Variable(torch.zeros(max_encoder_length, batch_size, self.max_length), requires_grad=False)
        attn_weights = attn_weights.cuda() if self.use_cuda else attn_weights
        output_variable = None
        if not self.training:
            output_variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
            output_variable = output_variable.cuda() if self.use_cuda else output_variable

        max_length = max(gtruth_lengths) if self.training else self.max_length

        for t in range(max_length):
            context, decoder_hidden, attn = decoder(current_input, decoder_hidden, encoder_output, context)
            decoder_output[t] = context
            attn_weights[t] = attn
            if self.training:
                current_input = gtruth_variable[t]
            else:
                scores = generator(context.squeeze(0))
                top_indices = scores.topk(1, dim=1)[1].view(-1)
                output_variable[t] = top_indices
                current_input = top_indices
        if not self.training:
            output_variable = output_variable.detach()

        return RunResult(encoder_output, decoder_output, attn_weights, output_variable)