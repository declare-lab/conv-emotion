
def update_classification_losses(losses, nums, name, bs, loss):
    if not isinstance(loss, float):
        print(type(loss))
        raise

    nums[name] += bs

    losses[name] += loss * bs


def update_generation_losses(losses, nums, micro, macro, bs, length, loss):
    # Update Losses
    nums[macro] += bs

    if isinstance(length, int):
        update_indiv_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        update_tensor_generation_losses(
            losses, nums, micro, macro, bs, length, loss)


def update_indiv_generation_losses(losses, nums, micro,
                                   macro, bs, length, loss):
    nums[micro] += bs * length

    batch_loss = loss * bs

    losses[micro] += batch_loss
    losses[macro] += batch_loss / length


def update_tensor_generation_losses(losses, nums, micro,
                                    macro, bs, length, loss):
    nums[micro] += length.sum().item()

    losses[micro] += loss.sum().item()
    losses[macro] += (loss / length.float()).sum().item()
