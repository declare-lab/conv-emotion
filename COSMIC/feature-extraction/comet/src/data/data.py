import os
import comet.src.data.atomic as atomic_data
import comet.src.data.conceptnet as conceptnet_data
import comet.src.data.config as cfg

import comet.utils.utils as utils

import pickle
import torch
import json


start_token = "<START>"
end_token = "<END>"
blank_token = "<blank>"


def save_checkpoint(state, filename):
    print("Saving model to {}".format(filename))
    torch.save(state, filename)


def save_step(model, vocab, optimizer, opt, length, lrs):
    if cfg.test_save:
        name = "{}.pickle".format(utils.make_name(
            opt, prefix="garbage/models/", is_dir=False, eval_=True))
    else:
        name = "{}.pickle".format(utils.make_name(
            opt, prefix="models/", is_dir=False, eval_=True))
    save_checkpoint({
        "epoch": length, "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(), "opt": opt,
        "vocab": vocab, "epoch_learning_rates": lrs},
        name)


def save_eval_file(opt, stats, eval_type="losses", split="dev", ext="pickle"):
    if cfg.test_save:
        name = "{}/{}.{}".format(utils.make_name(
            opt, prefix="garbage/{}/".format(eval_type),
            is_dir=True, eval_=True), split, ext)
    else:
        name = "{}/{}.{}".format(utils.make_name(
            opt, prefix="results/{}/".format(eval_type),
            is_dir=True, eval_=True), split, ext)
    print("Saving {} {} to {}".format(split, eval_type, name))

    if ext == "pickle":
        with open(name, "wb") as f:
            pickle.dump(stats, f)
    elif ext == "txt":
        with open(name, "w") as f:
            f.write(stats)
    elif ext == "json":
        with open(name, "w") as f:
            json.dump(stats, f)
    else:
        raise


def load_checkpoint(filename, gpu=True):
    if os.path.exists(filename):
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage)
    else:
        print("No model found at {}".format(filename))
    return checkpoint


def make_data_loader(opt, *args):
    if opt.dataset == "atomic":
        return atomic_data.GenerationDataLoader(opt, *args)
    elif opt.dataset == "conceptnet":
        return conceptnet_data.GenerationDataLoader(opt, *args)


def set_max_sizes(data_loader, force_split=None):
    data_loader.total_size = {}
    if force_split is not None:
        data_loader.total_size[force_split] = \
            data_loader.sequences[force_split]["total"].size(0)
        return
    for split in data_loader.sequences:
        data_loader.total_size[split] = \
            data_loader.sequences[split]["total"].size(0)
