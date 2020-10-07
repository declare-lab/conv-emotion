import time
import numpy as np

import comet.src.train.batch as batch_utils
import comet.utils.utils as utils
import comet.src.evaluate.evaluate as base_evaluate


def make_evaluator(opt, *args, **kwargs):
    return ConceptNetGenerationEvaluator(opt, *args, **kwargs)


class ConceptNetGenerationEvaluator(base_evaluate.Evaluator):
    def __init__(self, opt, model, data_loader, track=False):
        super(ConceptNetGenerationEvaluator, self).__init__(
            opt, model, data_loader)

        if track:
            self.tracker = {"positive": [], "negative": []}
        else:
            self.tracker = None

    def batch(self, opt, nums, average_loss, batch_variables, eval_mode):
        batch_variables["category"] = self.current_category

        outputs = batch_utils.batch_conceptnet_generate(
            opt, nums, average_loss, batch_variables, eval_mode,
            tracking_mode=self.tracker is not None)

        if outputs.get("tracking", None) is not None:
            self.tracker[self.current_category] += outputs["tracking"]

        if outputs["reset"] and batch_variables["category"] == "positive":
            outputs["reset"] = False
            self.current_category = "negative"

        return outputs

    def initialize_losses(self):
        average_loss = {"total_micro": 0, "total_macro": 0,
                        "negative_micro": 0, "negative_macro": 0}
        nums = {"total_micro": 0, "total_macro": 0,
                "negative_micro": 0, "negative_macro": 0}

        self.current_category = "positive"

        if self.tracker is not None:
            self.tracker = {"positive": [], "negative": []}

        return average_loss, nums

    def compute_final_scores(self, average_loss, nums):
        average_loss["total_macro"] /= nums["total_macro"]
        average_loss["total_micro"] /= nums["total_micro"]

        if nums["negative_micro"]:
            average_loss["negative_macro"] /= nums["negative_macro"]
            average_loss["negative_micro"] /= nums["negative_micro"]
        else:
            average_loss["negative_macro"] = 0
            average_loss["negative_micro"] = 0

        average_loss["macro_diff"] = (average_loss["negative_macro"] -
                                      average_loss["total_macro"])
        average_loss["micro_diff"] = (average_loss["negative_micro"] -
                                      average_loss["total_micro"])

        average_loss["ppl_macro"] = np.exp(average_loss["total_macro"])
        average_loss["ppl_micro"] = np.exp(average_loss["total_micro"])

        return average_loss

    def counter(self, nums):
        return nums["total_macro"]

    def print_result(self, split, epoch_losses):
        print("{} Loss: \t {}".format(
            split, epoch_losses["total_micro"]))
        print("{} Diff: \t {}".format(
            split, epoch_losses["micro_diff"]))
        print("{} Perplexity: \t {}".format(
            split, epoch_losses["ppl_micro"]))
