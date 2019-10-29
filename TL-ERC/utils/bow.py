import numpy as np
from collections import Counter
import torch.nn as nn
from torch.nn import functional as F
import torch
from math import isnan
from .vocab import PAD_ID, EOS_ID


def to_bow(sentence, vocab_size):
    '''  Convert a sentence into a bag of words representation
    Args
        - sentence: a list of token ids
        - vocab_size: V
    Returns
        - bow: a integer vector of size V
    '''
    bow = Counter(sentence)
    # Remove EOS tokens
    bow[PAD_ID] = 0
    bow[EOS_ID] = 0

    x = np.zeros(vocab_size, dtype=np.int64)
    x[list(bow.keys())] = list(bow.values())

    return x


def bag_of_words_loss(bow_logits, target_bow, weight=None):
    ''' Calculate bag of words representation loss
    Args
        - bow_logits: [num_sentences, vocab_size]
        - target_bow: [num_sentences]
    '''
    log_probs = F.log_softmax(bow_logits, dim=1)
    target_distribution = target_bow / (target_bow.sum(1).view(-1, 1) + 1e-23) + 1e-23
    entropy = -(torch.log(target_distribution) * target_bow).sum()
    loss = -(log_probs * target_bow).sum() - entropy

    return loss
