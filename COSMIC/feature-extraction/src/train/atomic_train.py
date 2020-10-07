import random

import comet.src.train.train as base_train
import comet.src.train.batch as batch
import comet.src.evaluate.atomic_evaluate as evaluate
# import comet.src.evaluate.atomic_generate as gen


def make_trainer(opt, *args):
    return AtomicGenerationIteratorTrainer(opt, *args)


class AtomicGenerationIteratorTrainer(base_train.IteratorTrainer):
    def __init__(self, opt, *args):
        super(AtomicGenerationIteratorTrainer, self).__init__(opt, *args)

        self.initialize_losses(opt.data.get("categories", []))

    def set_evaluator(self, opt, model, data_loader):
        self.evaluator = evaluate.make_evaluator(
            opt, model, data_loader)

    # def set_generator(self, opt, model, data_loader, scores, reward=None):
    #     self.generator = gen.make_generator(
    #         opt, model, data_loader, scores, reward)

    def set_sampler(self, opt):
        if opt.train.static.samp not in self.samplers:
            self.samplers[opt.train.static.samp] = sampling.make_sampler(
                opt.train.static.samp, opt, self.data_loader, batch_mode=True)
        self.batch_variables["sampler"] = self.samplers

    def batch(self, opt, *args):
        outputs = batch.batch_atomic_generate(opt, *args)

        token_loss = outputs["loss"]
        nums = outputs["nums"]
        reset = outputs["reset"]

        return token_loss, nums, reset

    def initialize_losses(self, categories):
        self.losses["train"] = {
            "total_micro": [0],
            "total_macro": [0]
        }

        nums = {"total_micro": 0, "total_macro": 0}

        for category in categories:
            micro_name = "{}_micro".format(category)
            macro_name = "{}_macro".format(category)

            self.losses["train"][micro_name] = [0]
            self.losses["train"][macro_name] = [0]

            nums[micro_name] = 0
            nums[macro_name] = 0

        return nums

    def update_top_score(self, opt):
        print(self.top_score)
        if self.top_score is None:
            self.top_score = (self.opt.train.dynamic.epoch,
                              self.get_tracked_score())
        elif self.get_tracked_score() < self.top_score[-1]:
            self.top_score = (self.opt.train.dynamic.epoch,
                              self.get_tracked_score())
        print(self.top_score)

    def get_tracked_score(self):
        return self.losses["dev"]["total_micro"][self.opt.train.dynamic.epoch]

    def counter(self, nums):
        return nums["total_macro"]
