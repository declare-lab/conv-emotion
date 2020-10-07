import time
import torch

import comet.utils.utils as utils
import comet.src.data.config as cfg


class Evaluator(object):
    def __init__(self, opt, model, data_loader):
        super(Evaluator, self).__init__()

        self.data_loader = data_loader
        self.model = model

        self.batch_variables = {
            "model": model,
            "data": data_loader
        }

        self.opt = opt

    def validate(self, l, split="dev", losses={}, keyset=None):
        self.batch_variables["split"] = split
        print("Evaluating {}".format(split))

        epoch_losses = self.epoch(
            self.opt, self.model, self.data_loader, split, keyset)

        self.print_result(split, epoch_losses)

        for loss_name, loss_val in epoch_losses.items():
            losses.setdefault(loss_name, {})
            losses[loss_name][l] = loss_val

    def epoch(self, opt, model, data_loader, split, keyset=None):
        average_loss, nums = self.initialize_losses()

        data_loader.reset_offsets(splits=split, shuffle=False)

        # Set evaluation mode
        model.eval()

        start = time.time()

        # Initialize progress bar
        bar = utils.set_progress_bar(
            data_loader.total_size[split])

        reset = False

        with torch.no_grad():
            while not reset:

                start = data_loader.offset_summary(split)

                outputs = self.batch(
                    opt, nums, average_loss,
                    self.batch_variables, eval_mode=True)

                end = data_loader.offset_summary(split)

                reset = outputs["reset"]

                if not reset:
                    bar.update(end - start)
                else:
                    print(end)

                if cfg.toy and self.counter(nums) > 100:
                    break
                if (opt.eval.es != "full" and
                        (self.counter(nums) > opt.eval.es)):
                    break

        nums = outputs["nums"]

        torch.cuda.synchronize()

        print("{} evaluation completed in: {} s".format(
            split.capitalize(), time.time() - start))

        average_loss = self.compute_final_scores(
            average_loss, nums)

        return average_loss
