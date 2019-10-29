import torch
from torch.nn import functional as F
import torch.nn as nn
from util import to_var, sequence_mask


# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()
