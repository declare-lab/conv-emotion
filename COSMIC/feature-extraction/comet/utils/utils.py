import json
import copy

import torch

import numpy as np
import contextlib

from distutils.dir_util import mkpath

from tqdm import tqdm


def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None,
              do_epoch=True):
    string = prefix
    string += "{}-{}".format(opt.dataset, opt.exp)
    string += "/"
    string += "{}-{}-{}".format(opt.trainer, opt.cycle, opt.iters)
    string += "/"
    string += opt.model
    if opt.mle:
        string += "-{}".format(opt.mle)
    string += "/"
    string += make_name_string(opt.data) + "/"

    string += make_name_string(opt.net) + "/"
    string += make_name_string(opt.train.static) + "/"

    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(
        opt.train.dynamic, True, do_epoch, set_epoch)
    if is_dir:
        mkpath(string)

    return string


def make_name_string(dict_, final=False, do_epoch=False, set_epoch=None):
    if final:
        if not do_epoch:
            string = "{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs)
        elif set_epoch is not None:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.items():
        if type(v) == DD:
            continue
        if isinstance(v, list):
            val = "#".join(is_bool(str(vv)) for vv in v)
        else:
            val = is_bool(v)
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string


def is_bool(v):
    if str(v) == "False":
        return "F"
    elif str(v) == "True":
        return "T"
    return v


def generate_config_files(type_, key, name="base", eval_mode=False):
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    # for param in changes[key]:
    #     base_config[param] = changes[key][param]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


def initialize_progress_bar(data_loader_list):
    num_examples = sum([len(tensor) for tensor in
                        data_loader_list.values()])
    return set_progress_bar(num_examples)


def set_progress_bar(num_examples):
    bar = tqdm(total=num_examples)
    bar.update(0)
    return bar


def merge_list_of_dicts(L):
    result = {}
    for d in L:
        result.update(d)
    return result


def return_iterator_by_type(data_type):
    if isinstance(data_type, dict):
        iterator = data_type.items()
    else:
        iterator = enumerate(data_type)
    return iterator


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def flatten(outer):
    return [el for inner in outer for el in inner]


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


# Taken from Jobman 0.1
class DD(dict):
    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
#         if attr.startswith('__'):
#             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z
