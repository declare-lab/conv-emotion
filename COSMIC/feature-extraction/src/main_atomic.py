
import random

import torch

import comet.src.train.atomic_train as train
import comet.src.models.models as models
import comet.src.data.data as data
import comet.utils.utils as utils
import comet.src.train.utils as train_utils
import comet.src.data.config as cfg

from comet.src.data.utils import TextEncoder
from comet.src.train.opt import OpenAIAdam


def main(num):
    # Generate configuration files depending on experiment being run
    utils.generate_config_files("atomic", num)

    # Loads the correct configuration file
    config_file = "config/atomic/config_{}.json".format(num)

    print(config_file)

    # Read config file to option
    config = cfg.read_config(cfg.load_config(config_file))
    opt, meta = cfg.get_parameters(config)

    # Set the random seeds
    torch.manual_seed(opt.train.static.seed)
    random.seed(opt.train.static.seed)
    if config.gpu_mode:
        torch.cuda.manual_seed_all(opt.train.static.seed)

    # Where to find the data
    splits = ["train", "dev", "test"]

    opt.train.dynamic.epoch = 0

    print("Loading Data")

    categories = opt.data.categories

    path = "data/atomic/processed/{}/{}.pickle".format(
        opt.exp, utils.make_name_string(opt.data))

    data_loader = data.make_data_loader(opt, categories)
    loaded = data_loader.load_data(path)
    print(data_loader.sequences["train"]["total"].size(0))
    data_loader.opt = opt
    data_loader.batch_size = opt.train.dynamic.bs

    print("Done.")

    # Initialize text_encoder
    text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in categories]
    special += [data.blank_token]

    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    opt.data.maxe1 = data_loader.max_event
    opt.data.maxe2 = data_loader.max_effect
    opt.data.maxr = data.atomic_data.num_delimiter_tokens["category"]

    n_special = len(special)
    n_ctx = opt.data.maxe1 + opt.data.maxe2
    n_vocab = len(text_encoder.encoder) + n_ctx

    print(data_loader.__dict__.keys())
    opt.net.vSize = n_vocab

    print("Building Model")

    model = models.make_model(
        opt, n_vocab, n_ctx, n_special,
        load=(opt.net.init=="pt"))

    print("Done.")

    print("Files will be logged at: {}".format(
        utils.make_name(opt, prefix="results/losses/",
                        is_dir=True, eval_=True)))

    data_loader.reset_offsets("train")

    # Get number of examples
    data.set_max_sizes(data_loader)

    if config.gpu_mode:
        print("Pushing to GPU: {}".format(config.gpu_index))
        cfg.device = config.gpu_index
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        if config.multigpu:
            model = models.multi_gpu(
                model, config.gpu_indices).cuda()
        else:
            model.cuda(cfg.device)
        print("Done.")

    print("Training")

    optimizer = OpenAIAdam(model.parameters(),
                           lr=opt.train.dynamic.lr,
                           schedule=opt.train.static.lrsched,
                           warmup=opt.train.static.lrwarm,
                           t_total=meta.iterations,
                           b1=opt.train.static.b1,
                           b2=opt.train.static.b2,
                           e=opt.train.static.e,
                           l2=opt.train.static.l2,
                           vector_l2=opt.train.static.vl2,
                           max_grad_norm=opt.train.static.clip)

    scorers = ["bleu", "rouge", "cider"]
    trainer = train.make_trainer(
        opt, meta, data_loader, model, optimizer)
    trainer.set_evaluator(opt, model, data_loader)

    trainer.run()
