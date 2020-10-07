import comet.src.data.data as data
import comet.src.data.config as cfg
import comet.src.evaluate.sampler as sampling


def do_gen_run(opt, generator, l, split="dev", scores={}):
    # Generate sequences for examples in evaluation set using
    # current trained model

    if opt.eval.gs == "full":
        sequences, avg_scores, indiv_scores = generator.generate(split)
    else:
        sequences, avg_scores, indiv_scores = generator.generate_some(split)

    if avg_scores is not None:
        # Record scores from generated sequences
        for score_name, score_val in avg_scores.items():
            scores.setdefault(score_name, {})
            scores[score_name].setdefault(l, [])
            scores[score_name][l] += [score_val]

    # Save generated sequences
    save_sequences(opt, sequences, avg_scores, indiv_scores,
                   l, split, opt.eval.gs == "full",
                   generator.data_loader)


def save_sequences(opt, sequences, avg_scores, indiv_scores,
                   l, split, full, data_loader):
    # This seems a bit roundabout since l = opt.train.dynamic in train.py
    # But it's in case we start checkpointing outside of epoch boundaries
    opt.train.dynamic.epoch = l

    if cfg.save:
        if full:
            names = {"gens": "gens", "scores": "scores",
                     "indiv": "indiv.scores"}
        else:
            names = {"gens": "gens.small", "scores": "scores.small",
                     "indiv": "indiv.scores.small"}
        # Save generated sequences
        data.save_eval_file(opt, sequences, names["gens"], split)

        if avg_scores is not None:
            # Save average scores over evaluation set for generated sequences
            # Scores computed are the ones the generator was initialized with
            data.save_eval_file(opt, avg_scores, names["scores"], split)

            if split == "dev":
                # Save individual scores
                data.save_eval_file(
                    opt, indiv_scores, names["indiv"], split)


class Generator(object):
    def __init__(self, opt, model, data_loader, scorers, reward_function=None):
        super(Generator, self).__init__()
        self.opt = opt

        self.model = model
        self.data_loader = data_loader

        self.sampler = sampling.make_sampler(
            opt.eval.sample, opt, data_loader)


    def generate(self, split="dev"):
        pass

    def generate_batch(self, sequences, split, verbose=False, bs=32):
        pass

