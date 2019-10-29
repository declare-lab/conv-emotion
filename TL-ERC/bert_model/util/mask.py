import torch
from .convert import to_var


def sequence_mask(sequence_length, max_len=None):
    """
    Args:
        sequence_length (Variable, LongTensor) [batch_size]
            - list of sequence length of each batch
        max_len (int)
    Return:
        masks (bool): [batch_size, max_len]
            - True if current sequence is valid (not padded), False otherwise

    Ex.
    sequence length: [3, 2, 1]

    seq_length_expand
    [[3, 3, 3],
     [2, 2, 2]
     [1, 1, 1]]

    seq_range_expand
    [[0, 1, 2]
     [0, 1, 2],
     [0, 1, 2]]

    masks
    [[True, True, True],
     [True, True, False],
     [True, False, False]]
    """
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)

    # [max_len]
    seq_range = torch.arange(0, max_len).long()  # [0, 1, ... max_len-1]

    # [batch_size, max_len]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = to_var(seq_range_expand)

    # [batch_size, max_len]
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    # [batch_size, max_len]
    masks = seq_range_expand < seq_length_expand

    return masks
