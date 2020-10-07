import torch
import torch.nn as nn
import torch.nn.functional as F

import comet.src.data.config as cfg
import comet.src.data.data as data
import comet.src.train.utils as train_utils
import comet.src.train.batch as batch

import comet.src.evaluate.evaluate as evaluate
import comet.src.evaluate.generate as gen
import comet.src.evaluate.sampler as sampling

import comet.utils.utils as utils

from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, opt, meta, data_loader, model, optimizer):
        self.optimizer = optimizer

        self.model = model

        if opt.trainer == "epoch":
            self.epochs = meta.epochs
        self.data_loader = data_loader
        self.opt = opt

        self.losses = {"dev": {}, "test": {}, "train": {}}

        self.top_score = None

        self.lrs = {}

        self.batch_variables = {
            "data": self.data_loader,
            "model": self.model,
            "split": "train"
        }

        self.do_gen = cfg.do_gen
        self.samplers = {}

    def decide_to_save(self):
        to_save = cfg.save and not cfg.toy

        to_save = to_save or cfg.test_save
        print(cfg.save_strategy)
        if cfg.save_strategy == "best":
            if self.top_score[0] != self.opt.train.dynamic.epoch:
                print("DOING IT RIGHT")
                to_save = False
        return to_save

    def save_model(self, tracked_score):
        lrs = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs[i] = param_group['lr']
        self.lrs[self.opt.train.dynamic.epoch] = lrs

        to_save = self.decide_to_save()

        if to_save:
            data.save_step(
                self.model, self.data_loader.vocab_encoder,
                self.optimizer, self.opt,
                self.opt.train.dynamic.epoch, self.lrs)

    def log_losses(self, opt, losses):
        if (not cfg.toy and cfg.save) or cfg.test_save:
            data.save_eval_file(opt, losses["train"], "losses", split="train")
            data.save_eval_file(opt, losses['dev'], "losses", split="dev")
            data.save_eval_file(opt, losses['test'], "losses", split="test")

    def set_logger(self):
        if cfg.toy:
            self.logger = SummaryWriter(utils.make_name(
                self.opt, prefix="garbage/logs/", eval_=True, do_epoch=False))
        else:
            self.logger = SummaryWriter(utils.make_name(
                self.opt, prefix="logs/", eval_=True, do_epoch=False))
        print("Logging Tensorboard Files at: {}".format(self.logger.logdir))

    def stop_logger(self):
        self.logger.close()

    def run(self):
        self.set_logger()
        self.count = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.opt.train.dynamic.epoch += 1
            self.epoch()

        self.stop_logger()

    def epoch(self):
        nums = self.reset_losses()

        # Initialize progress bar
        bar = utils.initialize_progress_bar(
            self.data_loader.sequences["train"])

        reset = False

        while not reset:
            loss, nums, reset = self.do_forward_pass(nums)
            self.do_backward_pass(loss)
            self.update_parameters()

            bar.update(self.opt.train.dynamic.bs)
            self.count += 1

            for loss_name in self.losses["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss.item() / self.opt.train.dynamic.bs,
                    self.count)

            if cfg.toy and self.counter(nums) > 300:
                break

        with torch.no_grad():
            self.run_evaluation_cycle()

        self.log_losses(self.opt, self.losses)
        self.update_top_score(self.opt)
        self.save_model(self.get_tracked_score())

        self.data_loader.reset_offsets("train")

    def run_evaluation_cycle(self):
        for split in ["dev", "test"]:
            self.evaluator.validate(
                self.opt.train.dynamic.epoch, split,
                self.losses[split])

            if self.do_gen:
                gen.do_gen_run(
                    self.opt, self.generator, self.opt.train.dynamic.epoch,
                    split, self.losses[split])
            iter_num = self.opt.train.dynamic.epoch

            for loss_name in self.losses[split]:
                self.logger.add_scalar(
                    "{}/{}".format(split, loss_name),
                    self.losses[split][loss_name][iter_num],
                    iter_num)

    def clip_gradients(self):
        if self.opt.train.static.clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.train.static.clip)

    def do_forward_pass(self, nums):
        token_loss, nums, reset = self.batch(
            self.opt, nums, self.losses["train"],
            self.batch_variables)
        return token_loss, nums, reset

    def do_backward_pass(self, loss):
        loss.backward()

    def update_parameters(self):
        if self.opt.model == "lstm":
                self.clip_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reset_losses(self):
        loss_names = set([i.rstrip("maicro").rstrip("_") for
                          i in self.losses["train"].keys()])
        return self.initialize_losses(list(loss_names))


class IteratorTrainer(Trainer):
    def __init__(self, opt, meta, data_loader, model, optimizer):
        super(IteratorTrainer, self).__init__(
            opt, meta, data_loader, model, optimizer)

        self.iters = meta.cycle
        self.total_iters = meta.iterations

    def run(self):
        self.set_logger()

        # Initialize progress bar
        bar = utils.set_progress_bar(self.total_iters)

        for cycle_num in range(int(self.total_iters / self.iters)):
            self.model.train()

            self.cycle(bar, cycle_num)

            with torch.no_grad():
                self.run_evaluation_cycle()

            self.log_losses(self.opt, self.losses)
            self.update_top_score(self.opt)
            self.save_model(self.get_tracked_score())

        self.stop_logger()

    def cycle(self, bar, cycle_num):
        nums = self.reset_losses()
        print(self.losses["train"])

        for i in range(1, self.iters + 1):
            # self.model.zero_grad()

            loss, nums, reset = self.do_forward_pass(nums)
            self.do_backward_pass(loss)

            self.update_parameters()
            # print(loss)
            # print(loss.item())
            self.opt.train.dynamic.epoch += 1

            for loss_name in self.losses["train"]:
                self.logger.add_scalar(
                    "train/{}".format(loss_name),
                    loss.item() / self.opt.train.dynamic.bs,
                    self.opt.train.dynamic.epoch)

            bar.update(1)

            if cfg.toy and i > 10:
                break

            if reset:
                self.data_loader.reset_offsets("train")

