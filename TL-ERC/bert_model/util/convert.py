import torch
from torch.autograd import Variable


def to_var(x, on_cpu=False, gpu_id=None, async=False):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async)
        #x = Variable(x)
    return x


def to_tensor(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def reverse_order(tensor, dim=0):
    """Reverse Tensor or Variable"""
    if isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.LongTensor):
        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = tensor.index_select(dim, idx)
    if isinstance(tensor, torch.cuda.FloatTensor) or isinstance(tensor, torch.cuda.LongTensor):
        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]
        idx = torch.cuda.LongTensor(idx)
        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor
    elif isinstance(tensor, Variable):
        variable = tensor
        variable.data = reverse_order(variable.data, dim)
        return variable

def reverse_order_valid(tensor, length_list, dim=0):
    """
    Reverse Tensor of Variable only in given length
    Ex)
    Args:
        - tensor (Tensor or Variable)
         1   2   3   4   5   6
         6   7   8   9   0   0
        11  12  13   0   0   0
        16  17   0   0   0   0
        21  22  23  24  25  26

        - length_list (list)
        [6, 4, 3, 2, 6]
 
    Return:
        tensor (Tensor or Variable; in-place)
         6   5   4   3   2   1
         0   0   9   8   7   6
         0   0   0  13  12  11
         0   0   0   0  17  16
        26  25  24  23  22  21
    """
    for row, length in zip(tensor, length_list):
        valid_row = row[:length]
        reversed_valid_row = reverse_order(valid_row, dim=dim)
        row[:length] = reversed_valid_row
    return tensor
