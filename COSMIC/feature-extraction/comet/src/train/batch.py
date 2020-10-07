
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import comet.src.data.config as cfg
import comet.src.train.utils as train_utils
import comet.src.models.utils as model_utils
import comet.src.evaluate.utils as eval_utils
import comet.utils.utils as utils
from IPython import embed


##############################################################################
#                                       BATCH
##############################################################################


def batch_atomic_generate(opt, nums, losses, batch_variables, eval_mode=False):
    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    split = batch_variables["split"]

    batch, reset = data_loader.sample_batch(split, bs=opt.train.dynamic.bs)

    input_ = model_utils.prepare_position_embeddings(
        opt, data_loader.vocab_encoder, batch["sequences"].unsqueeze(-1))
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]

    targets = input_.squeeze(0)[:, 1:, 0].contiguous().view(-1)

    loss, dist = mle_steps(
        opt.net.model, model, input_[:, :-1, :], targets,
        attention_mask[:, :-1], loss_reduction="none")

    # Set loss name
    micro_name = "total_micro"
    macro_name = "total_macro"

    length = loss_mask.sum(1)
    bs = input_.size(0)

    final_loss = (loss * loss_mask).sum(1)

    update_generation_losses(losses, nums, micro_name, macro_name, bs,
                             length, (loss * loss_mask).sum(1), split)

    final_loss = final_loss / length

    outputs = {"loss": final_loss.sum(), "nums": nums, "reset": reset}

    return outputs


def batch_conceptnet_generate(opt, nums, losses, batch_variables,
                              eval_mode=False, tracking_mode=False):
    data_loader = batch_variables["data"]
    model = batch_variables["model"]
    split = batch_variables["split"]
    category = batch_variables["category"]

    batch, reset = data_loader.sample_batch(
        split, bs=opt.train.dynamic.bs, cat=category)

    input_ = model_utils.prepare_position_embeddings(
        opt, data_loader.vocab_encoder, batch["sequences"].unsqueeze(-1))
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]

    targets = input_.squeeze(0)[:, 1:, 0].contiguous().view(-1)

    loss, dist = mle_steps(
        opt.net.model, model, input_[:, :-1, :], targets,
        attention_mask[:, :-1], loss_reduction="none")

    # Set loss name
    if not eval_mode or batch_variables["category"] == "positive":
        micro_name = "total_micro"
        macro_name = "total_macro"
    else:
        micro_name = "negative_micro"
        macro_name = "negative_macro"

    length = loss_mask.sum(1)
    bs = input_.size(0)

    final_loss = (loss * loss_mask).sum(1)

    update_generation_losses(losses, nums, micro_name, macro_name, bs,
                             length, (loss * loss_mask).sum(1), split)

    final_loss = final_loss / length

    outputs = {"loss": final_loss.sum(), "nums": nums, "reset": reset}

    if tracking_mode:
        outputs["tracking"] = final_loss.squeeze().tolist()

    return outputs


def mle_steps(key, model, input_, targets, attention_mask,
              loss_reduction="mean", i=None):
    word_acts = decode(model, input_.unsqueeze(1),
                       attention_mask, i)

    word_dist = train_utils.modify_output_for_loss_fn(
        "nll", word_acts, dim=-1)

    # Compute losses
    loss = F.nll_loss(
        word_dist.view(-1, word_dist.size(-1)),
        targets, reduction=loss_reduction)

    if loss_reduction != "mean":
        return loss.view(word_dist.size(0), -1), word_dist
    else:
        return loss, word_dist


def decode(model, input_, attention_mask, i=None):
    return model(input_, sequence_mask=attention_mask)


def update_generation_losses(losses, nums, micro, macro, bs,
                             length, loss, split):
    if split == "train":
        train_utils.update_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        eval_utils.update_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
