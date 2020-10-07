from comet.src.models.gpt import (LMModel, DEFAULT_CONFIG, load_openai_pretrained_model)
import torch.nn as nn


def make_model(opt, n_vocab, n_ctx, n_special, load=True,
               return_acts=True, return_probs=False,
               clf_token="<CLASS>", answer_size=None):
    print(n_ctx)
    if opt.exp == "generation":
        model = LMModel(
            opt.net, n_vocab, n_ctx, return_acts=return_acts,
            return_probs=return_probs)
    elif opt.exp == "classification":
        model = ClfModel(
            opt.net, n_vocab, n_ctx, clf_token, answer_size)
    if load:
        print("LOADING PRETRAINED TRANSFORMER")
        load_openai_pretrained_model(
            model.transformer, n_ctx=n_ctx, n_special=n_special)
    return model


def multi_gpu(model, devices):
    return nn.DataParallel(model, device_ids=devices)


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {i[len("module."):]: j for i, j in state_dict.items()}
        model.load_state_dict(new_state_dict)
