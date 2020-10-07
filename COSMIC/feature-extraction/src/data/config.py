import json
from comet.utils.utils import DD

device = "cpu"

save = False
test_save = False
toy = False
do_gen = False

save_strategy = "all"


def get_parameters(opt, exp_type="model"):
    params = DD()
    params.net = DD()

    params.mle = 0
    params.dataset = opt.dataset

    params.net = get_net_parameters(opt)
    params.train = get_training_parameters(opt)

    params.model = params.net.model
    params.exp = opt.exp

    params.data = get_data_parameters(opt, params.exp, params.dataset)
    params.eval = get_eval_parameters(opt, params.data.get("categories", None))

    meta = DD()

    params.trainer = opt.trainer

    meta.iterations = int(opt.iterations)
    meta.cycle = opt.cycle
    params.cycle = opt.cycle
    params.iters = int(opt.iterations)

    global toy
    toy = opt.toy

    global do_gen
    do_gen = opt.do_gen

    global save
    save = opt.save

    global test_save
    test_save = opt.test_save

    global save_strategy
    save_strategy = opt.save_strategy

    print(params)
    return params, meta


def get_eval_parameters(opt, force_categories=None):
    evaluate = DD()

    if opt.eval_sampler == "beam":
        evaluate.bs = opt.beam_size
    elif opt.eval_sampler == "greedy":
        evaluate.bs = 1
    elif opt.eval_sampler == "topk":
        evaluate.k = opt.topk_size

    evaluate.smax = opt.gen_seqlength
    evaluate.sample = opt.eval_sampler

    evaluate.numseq = opt.num_sequences

    evaluate.gs = opt.generate_sequences
    evaluate.es = opt.evaluate_sequences

    if opt.dataset == "atomic":
        if "eval_categories" in opt and force_categories is None:
            evaluate.categories = opt.eval_categories
        else:
            evaluate.categories = force_categories

    return evaluate


def get_data_parameters(opt, experiment, dataset):
    data = DD()
    if dataset == "atomic":
        data.categories = sorted(opt.categories)
        # hard-coded
        data.maxe1 = 17
        data.maxe2 = 35
        data.maxr = 1

    elif dataset == "conceptnet":
        data.rel = opt.relation_format
        data.trainsize = opt.training_set_size
        data.devversion = opt.development_set_versions_to_use
        data.maxe1 = opt.max_event_1_size
        data.maxe2 = opt.max_event_2_size
        if data.rel == "language":
            # hard-coded
            data.maxr = 5
        else:
            # hard-coded
            data.maxr = 1

    return data


def get_training_parameters(opt):
    train = DD()
    static = DD()
    static.exp = opt.exp

    static.seed = opt.random_seed

    # weight decay
    static.l2 = opt.l2
    static.vl2 = True
    static.lrsched = opt.learning_rate_schedule  # 'warmup_linear'
    static.lrwarm = opt.learning_rate_warmup  # 0.002

    # gradient clipping
    static.clip = opt.clip

    # what loss function to use
    static.loss = opt.loss

    dynamic = DD()
    dynamic.lr = opt.learning_rate  # learning rate
    dynamic.bs = opt.batch_size  # batch size
    # optimizer to use {adam, rmsprop, etc.}
    dynamic.optim = opt.optimizer

    # rmsprop
    # alpha is interpolation average

    static.update(opt[dynamic.optim])

    train.static = static
    train.dynamic = dynamic

    return train


def get_net_parameters(opt):
    net = DD()
    net.model = opt.model
    net.nL = opt.num_layers
    net.nH = opt.num_heads
    net.hSize = opt.hidden_dim
    net.edpt = opt.embedding_dropout
    net.adpt = opt.attention_dropout
    net.rdpt = opt.residual_dropout
    net.odpt = opt.output_dropout
    net.pt = opt.pretrain
    net.afn = opt.activation

    # how to intialize parameters
    # format is gauss+{}+{}.format(mean, std)
    # n = the default initialization pytorch
    net.init = opt.init

    return net


def read_config(file_):
    config = DD()
    print(file_)
    for k, v in file_.items():
        if v == "True" or v == "T" or v == "true":
            config[k] = True
        elif v == "False" or v == "F" or v == "false":
            config[k] = False
        elif type(v) == dict:
            config[k] = read_config(v)
        else:
            config[k] = v

    return config


def load_config(name):
    with open(name, "r") as f:
        config = json.load(f)
    return config
