import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, n_layers=3, dropout=0.3):
        super(EncoderRNN, self).__init__()

        num_directions = 2
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)

        packed = pack(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = unpack(outputs)

        n = hidden[0].size(0)
        hidden = (torch.cat([hidden[0][0:n:2], hidden[0][1:n:2]], 2), torch.cat([hidden[1][0:n:2], hidden[1][1:n:2]], 2))
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, n_layers=3,
                 dropout=0.3, max_length=50, use_cuda=False):
        super(DecoderRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout)

    def step(self, input_seq, hidden):
        # hidden: S = n_layers x B x N
        embedded = self.embedding(input_seq).unsqueeze(0)  # S = 1 x B x E
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_state(self, batch_size, sos_index):
        initial_input = Variable(torch.zeros((batch_size, )).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, inputs, input_lengths, hidden, initial_input, one_step=False):
        current_input = initial_input
        if one_step:
            output, hidden = self.step(current_input, hidden)
            return output, hidden

        batch_size = len(input_lengths)
        max_input_length = max(input_lengths) + 1
        outputs = Variable(torch.zeros(max_input_length, batch_size, self.hidden_size), requires_grad=False)
        outputs = outputs.cuda() if self.use_cuda else outputs
        for t in range(max_input_length):
            if t != 0:
                current_input = inputs[t-1]
            output, hidden = self.step(current_input, hidden)
            outputs[t] = output
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

    def step(self, input_seq, hidden, encoder_outputs, context):
        # hidden: S = n_layers x B x N
        # encoder_outputs: S = L x B x N
        embedded = self.embedding(input_seq).unsqueeze(0)  # S = 1 x B x E

        # Combine embedded input word and attended context, run through RNN (input feeding)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.rnn(rnn_input, hidden)

        # Calculate attention weights and apply to encoder outputs
        output, attn_weights = self.attn(output, encoder_outputs)
        assert output.size(2) == self.hidden_size
        # output: # S = 1 x B x N

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def init_state(self, encoder_outputs, sos_index):
        max_encoder_length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        initial_input = Variable(torch.zeros((batch_size,)).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input

        attn_weights = Variable(torch.zeros(batch_size, max_encoder_length)).unsqueeze(1)  # S = B x 1 x L
        attn_weights = torch.add(attn_weights, 1.0 / max_encoder_length)
        attn_weights = attn_weights.cuda() if self.use_cuda else attn_weights
        initial_context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)  # S = 1 x B x N
        initial_context = initial_context.cuda() if self.use_cuda else initial_context
        initial_context = initial_context.detach()

        return initial_input, initial_context

    def forward(self, inputs, input_lengths, hidden, encoder_outputs, initial_input, initial_context, one_step=False):
        max_encoder_length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        context = initial_context
        current_input = initial_input

        if one_step:
            context, hidden, attn = self.step(current_input, hidden, encoder_outputs, context)
            return context, hidden, attn

        max_input_length = max(input_lengths) + 1

        outputs = Variable(torch.zeros(max_input_length, batch_size, self.hidden_size), requires_grad=False)
        outputs = outputs.cuda() if self.use_cuda else outputs
        attn_weights = Variable(torch.zeros(max_input_length, batch_size, max_encoder_length), requires_grad=False)
        attn_weights = attn_weights.cuda() if self.use_cuda else attn_weights

        for t in range(max_input_length):
            if t != 0:
                current_input = inputs[t-1] 
            context, hidden, attn = self.step(current_input, hidden, encoder_outputs, context)
            outputs[t] = context
            attn_weights[t] = attn

        return outputs, hidden, attn_weights
