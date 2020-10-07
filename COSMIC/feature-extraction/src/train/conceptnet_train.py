import random
import torch

import comet.src.data.config as cfg

import comet.src.train.atomic_train as base_train
import comet.src.train.batch as batch_utils
import comet.src.evaluate.conceptnet_evaluate as evaluate
import comet.src.evaluate.conceptnet_generate as gen


def make_trainer(opt, *args):
    return ConceptNetGenerationIteratorTrainer(opt, *args)


class ConceptNetGenerationIteratorTrainer(
        base_train.AtomicGenerationIteratorTrainer):
    def set_evaluator(self, opt, model, data_loader):
        self.evaluator = evaluate.make_evaluator(
            opt, model, data_loader)

    def set_generator(self, opt, model, data_loader):
        self.generator = gen.make_generator(
            opt, model, data_loader)

    def batch(self, opt, *args):
        outputs = batch_utils.batch_atomic_generate(opt, *args)

        token_loss = outputs["loss"]
        nums = outputs["nums"]
        reset = outputs["reset"]

        return token_loss, nums, reset

    def update_top_score(self, opt):
        print(self.top_score)

        tracked_scores = self.get_tracked_score()

        if self.top_score is None:
            self.top_score = \
                self.top_score = {"epoch": {}, "score": {}}
            self.top_score["epoch"]["total_micro"] = self.opt.train.dynamic.epoch
            self.top_score["score"]["total_micro"] = tracked_scores["total_micro"]
        else:
            if tracked_scores["total_micro"] < self.top_score["score"]["total_micro"]:
                self.top_score["epoch"]["total_micro"] = self.opt.train.dynamic.epoch
                self.top_score["score"]["total_micro"] = tracked_scores["total_micro"]

        print(self.top_score)

    def get_tracked_score(self):
        return {
            "total_micro": self.losses["dev"]["total_micro"][self.opt.train.dynamic.epoch]
        }

    def decide_to_save(self):
        to_save = cfg.save and not cfg.toy

        curr_epoch = self.opt.train.dynamic.epoch

        to_save = to_save or cfg.test_save
        print(cfg.save_strategy)
        if cfg.save_strategy == "best":
            if ((self.top_score["epoch"]["total_micro"] != curr_epoch)):
                to_save = False
        return to_save
