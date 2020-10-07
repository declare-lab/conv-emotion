import torch
import torch.optim
import torch.nn.functional as F

import copy


def update_generation_losses(losses, nums, micro, macro, bs, length, loss):
    # Update Losses
    losses[micro] += \
        [copy.deepcopy(losses[micro][-1])]
    losses[macro] += \
        [copy.deepcopy(losses[macro][-1])]

    losses[micro][-1] *= nums[micro]
    losses[macro][-1] *= nums[macro]

    nums[macro] += bs

    if isinstance(length, int):
        update_indiv_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        update_tensor_generation_losses(
            losses, nums, micro, macro, bs, length, loss)


def update_indiv_generation_losses(losses, nums, micro,
                                   macro, bs, length, loss):
    nums[micro] += (bs * length)

    batch_loss = loss * bs

    losses[micro][-1] += batch_loss
    losses[micro][-1] /= nums[micro]
    losses[macro][-1] += batch_loss / length
    losses[macro][-1] /= nums[macro]


def update_tensor_generation_losses(losses, nums, micro,
                                    macro, bs, length, loss):
    nums[micro] += length.sum().item()

    losses[micro][-1] += loss.sum().item()
    losses[micro][-1] /= nums[micro]
    losses[macro][-1] += (loss / length.float()).sum().item()
    losses[macro][-1] /= nums[macro]


def modify_output_for_loss_fn(loss_fn, output, dim):
    if loss_fn == "ce":
        return output
    if loss_fn == "mse":
        return F.softmax(output, dim=dim)
    if loss_fn == "nll":
        return F.log_softmax(output, dim=dim)
    if loss_fn in ["bce", "wbce", "wbce1"]:
        return torch.sigmoid(output)
