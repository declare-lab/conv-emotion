from collections import defaultdict
import pickle
import torch
import numpy as np
import os
from torch import Tensor
from torch.autograd import Variable
from nltk import FreqDist
from .convert import to_tensor, to_var


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


class Vocab(object):
    def __init__(self, tokenizer=None, max_size=None, min_freq=1):
        """Basic Vocabulary object"""

        self.vocab_size = 0
        self.freqdist = FreqDist()
        self.tokenizer = tokenizer

    def update(self, glove_dir, max_size=None, min_freq=1):
        """
        Initialize id2word & word2id based on self.freqdist
        max_size include 4 special tokens
        """

        # {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.id2word = {
            PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN,
            SOS_ID: SOS_TOKEN, EOS_ID: EOS_TOKEN
        }
        # {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.word2id = defaultdict(lambda: UNK_ID)  # Not in vocab => return UNK
        self.word2id.update({
            PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
            SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        })
        # self.word2id = {
        #     PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        #     SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        # }

        vocab_size = 4
        min_freq = max(min_freq, 1)

        # Reset frequencies of special tokens
        # [...('<eos>', 0), ('<pad>', 0), ('<sos>', 0), ('<unk>', 0)]
        freqdist = self.freqdist.copy()
        special_freqdist = {token: freqdist[token]
                            for token in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]}
        freqdist.subtract(special_freqdist)

        # Sort: by frequency, then alphabetically
        # Ex) freqdist = { 'a': 4,   'b': 5,   'c': 3 }
        #  =>   sorted = [('b', 5), ('a', 4), ('c', 3)]
        sorted_frequency_counter = sorted(freqdist.items(), key=lambda k_v: k_v[0])
        sorted_frequency_counter.sort(key=lambda k_v: k_v[1], reverse=True)

        # Load glove vector
        word_emb_dict = self.get_glove_emb(glove_dir)

        for word, freq in sorted_frequency_counter:

            if freq < min_freq or vocab_size == max_size:
                break
            self.id2word[vocab_size] = word
            self.word2id[word] = vocab_size
            vocab_size += 1

        self.vocab_size = vocab_size


        # Create embedding matrix
        self.embedding_matrix = embedding_matrix = np.zeros((self.vocab_size, 300))

        for word, ind in self.word2id.items():
            if word.lower() in word_emb_dict:
                embedding_matrix[self.word2id[word]] = word_emb_dict[word.lower()]
            else:
                embedding_matrix[self.word2id[word]] = np.random.uniform(-0.25, 0.25, 300)

    def get_glove_emb(self, GLOVE_DIR):
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), 'rb')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word.decode().lower()] = coefs
        f.close()
        return embeddings_index


    def __len__(self):
        return len(self.id2word)


    def load(self, word2id_path=None, id2word_path=None, word_emb_path=None):
        if word2id_path:
            with open(word2id_path, 'rb') as f:
                word2id = pickle.load(f)
            # Can't pickle lambda function
            self.word2id = defaultdict(lambda: UNK_ID)
            self.word2id.update(word2id)
            self.vocab_size = len(self.word2id)

        if id2word_path:
            with open(id2word_path, 'rb') as f:
                id2word = pickle.load(f)
            self.id2word = id2word
        
        if word_emb_path:
            with open(word_emb_path, 'rb') as f:
                embedding_matrix = pickle.load(f)
            self.embedding_matrix = embedding_matrix

    def add_word(self, word):
        assert isinstance(word, str), 'Input should be str'
        self.freqdist.update([word])

    def add_sentence(self, sentence, tokenized=False):
        if not tokenized:
            sentence = self.tokenizer(sentence)
        for word in sentence:
            self.add_word(word)

    def add_dataframe(self, conversation_df, tokenized=True):
        for conversation in conversation_df:
            for sentence in conversation:
                self.add_sentence(sentence, tokenized=tokenized)

    def pickle(self, word2id_path, id2word_path, word_emb_path):
        with open(word2id_path, 'wb') as f:
            pickle.dump(dict(self.word2id), f)

        with open(id2word_path, 'wb') as f:
            pickle.dump(self.id2word, f)

        with open(word_emb_path, 'wb') as f:
            pickle.dump(self.embedding_matrix, f)

    def to_list(self, list_like):
        """Convert list-like containers to list"""
        if isinstance(list_like, list):
            return list_like

        if isinstance(list_like, Variable):
            return list(to_tensor(list_like).numpy())
        elif isinstance(list_like, Tensor):
            return list(list_like.numpy())

    def id2sent(self, id_list):
        """list of id => list of tokens (Single sentence)"""
        id_list = self.to_list(id_list)
        sentence = []
        for id in id_list:
            word = self.id2word[id]
            if word not in [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]:
                sentence.append(word)
            if word == EOS_TOKEN:
                break
        return sentence

    def sent2id(self, sentence, var=False):
        """list of tokens => list of id (Single sentence)"""
        id_list = [self.word2id[word] for word in sentence]
        if var:
            id_list = to_var(torch.LongTensor(id_list), eval=True)
        return id_list

    def decode(self, id_list):
        sentence = self.id2sent(id_list)
        return ' '.join(sentence)
