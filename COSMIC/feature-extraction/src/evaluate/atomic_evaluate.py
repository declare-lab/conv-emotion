import comet.src.train.batch as batch
import comet.src.evaluate.evaluate as base_evaluate
import numpy as np

def make_evaluator(opt, *args):
    if opt.exp == "generation":
        return AtomicGenerationEvaluator(opt, *args)
    else:
        return AtomicClassificationEvaluator(opt, *args)


class AtomicGenerationEvaluator(base_evaluate.Evaluator):
    def __init__(self, opt, model, data_loader):
        super(AtomicGenerationEvaluator, self).__init__(
            opt, model, data_loader)

        self.batch = batch.batch_atomic_generate

    def initialize_losses(self):
        average_loss = {"total_micro": 0, "total_macro": 0}
        nums = {"total_micro": 0, "total_macro": 0}
        return average_loss, nums

    def compute_final_scores(self, average_loss, nums):
        average_loss["total_macro"] /= nums["total_macro"]
        average_loss["total_micro"] /= nums["total_micro"]

        average_loss["ppl_macro"] = np.exp(average_loss["total_macro"])
        average_loss["ppl_micro"] = np.exp(average_loss["total_micro"])

        return average_loss

    def counter(self, nums):
        return nums["total_macro"]

    def print_result(self, split, epoch_losses):
        print("{} Loss: \t {}".format(
            split, epoch_losses["total_micro"]))
        print("{} Perplexity: \t {}".format(
            split, epoch_losses["ppl_micro"]))
