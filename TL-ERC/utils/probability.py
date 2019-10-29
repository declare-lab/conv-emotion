import torch
import numpy as np
from .convert import to_var


def normal_logpdf(x, mean, var):
    """
    Args:
        x: (Variable, FloatTensor) [batch_size, dim]
        mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
        var: (Variable, FloatTensor) [batch_size, dim]: positive value
    Return:
        log_p: (Variable, FloatTensor) [batch_size]
    """

    pi = to_var(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - ((x - mean).pow(2) / var), dim=1)


def normal_kl_div(mu1, var1,
                  mu2=to_var(torch.FloatTensor([0.0])),
                  var2=to_var(torch.FloatTensor([1.0]))):
    one = to_var(torch.FloatTensor([1.0]))
    return torch.sum(0.5 * (torch.log(var2) - torch.log(var1)
                            + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)
