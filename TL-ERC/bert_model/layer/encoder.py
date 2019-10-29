import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from util import to_var, reverse_order_valid, PAD_ID
from .rnncells import StackedGRUCell, StackedLSTMCell

import copy


class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (to_var(torch.zeros(self.num_layers*self.num_directions,
                                       batch_size,
                                       self.hidden_size)),
                    to_var(torch.zeros(self.num_layers*self.num_directions,
                                       batch_size,
                                       self.hidden_size)))
        else:
            return to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True, batch_first=True, train_emb = False, emb_weights_matrix=None):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD_ID)
        # self.embedding.load_state_dict({'weight': to_var(torch.LongTensor(emb_weights_matrix))})
        # if (not train_emb):
        #     self.embedding.weight.requires_grad = False 

        self.rnn = rnn(input_size=embedding_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       bias=bias,
                       batch_first=batch_first,
                       dropout=dropout,
                       bidirectional=bidirectional)

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_seq_len]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.data.tolist()

        # [num_sentences, max_source_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_sentences, max_source_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(embedded, input_length_sorted,
                                         batch_first=self.batch_first)

        hidden = self.init_h(batch_size, hidden=hidden)
        

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        outputs, outputs_lengths = pad_packed_sequence(
            outputs, batch_first=self.batch_first)

        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                      hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden


class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.GRU, num_layers=1, dropout=0.0,
                 bidirectional=False, bias=True, batch_first=True):
        """Context-level Encoder"""
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size,
                       hidden_size=context_size,
                       num_layers=num_layers,
                       bias=bias,
                       batch_first=batch_first,
                       dropout=dropout,
                       bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_length, hidden=None):
        """
        Args:
            encoder_hidden (Variable, FloatTensor): [batch_size, max_len, num_layers * direction * hidden_size]
            conversation_length (Variable, LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_seq_len, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len, _ = encoder_hidden.size()

        # Sort for PackedSequence
        conv_length_sorted, indices = conversation_length.sort(descending=True)
        
        conv_length_sorted = conv_length_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        
        rnn_input = pack_padded_sequence(
            encoder_hidden_sorted, conv_length_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        # outputs: [batch_size, max_conversation_length, context_size]
        outputs, outputs_length = pad_packed_sequence(
            outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                      hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):

        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden
