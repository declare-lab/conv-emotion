import comet.src.data.utils as data_utils
import comet.src.data.atomic as adata
import comet.src.data.config as cfg

import torch
import random
from tqdm import tqdm


def map_name(name, opt):
    if name == "train":
        return "train{}k.txt".format(opt.trainsize)
    elif name == "test":
        return "test.txt"
    else:
        return "dev{}.txt".format(opt.devversion)


conceptnet_relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor'
]


split_into_words = {
    'AtLocation': "at location",
    'CapableOf': "capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "created by",
    'DefinedAs': "defined as",
    'DesireOf': "desire of",
    'Desires': "desires",
    'HasA': "has a",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'instance of',
    'IsA': "is a",
    'LocatedNear': "located near",
    'LocationOfAction': "location of action",
    'MadeOf': "made of",
    'MotivatedByGoal': "motivated by goal",
    'NotCapableOf': "not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "not made of",
    'PartOf': "part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "related to",
    'SymbolOf': "symbol of",
    'UsedFor': "used for"
}


class GenerationDataLoader(adata.DataLoader):
    def __init__(self, opt, categories=None):
        super(GenerationDataLoader, self).__init__(opt)
        self.opt = opt

        for split in self.data:
            self.data[split] = {"total": []}
            self.offsets[split] = {"total": 0}

        self.vocab_encoder = None
        self.vocab_decoder = None
        self.special_chars = None
        self.max_e1 = None
        self.max_e2 = None
        self.max_r = None

    def offset_summary(self, split):
        return sum(self.offsets[split].values())

    def load_data(self, path):
        if ".pickle" in path:
            print("Loading data from: {}".format(path))
            data_utils.load_existing_data_loader(self, path)
            return True

        for split in self.data:
            file_name = map_name(split, self.opt.data)

            if split != "dev" or self.opt.data.devversion != "12":
                string_tuples = open("{}/{}".format(
                    path, file_name), "r").read().split("\n")
                tuples = [x.split("\t") for x in string_tuples if x]
            else:
                string_tuples = open("{}/{}".format(
                    path, "dev1.txt"), "r").read().split("\n")
                tuples = [x.split("\t") for x in string_tuples if x]
                string_tuples = open("{}/{}".format(
                    path, "dev2.txt"), "r").read().split("\n")
                tuples += [x.split("\t") for x in string_tuples if x]

            if split in ["dev", "test"]:
                if self.opt.data.rel == "language":
                    self.data[split]["total"] = \
                        [(i[1].lower().strip(), split_into_words[i[0]],
                         i[2].lower().strip(), int(i[3])) for i in tuples]
                    self.data[split]["positive"] = \
                        [(i[1].lower().strip(), split_into_words[i[0]],
                         i[2].lower().strip(), int(i[3])) for i in tuples if int(i[3])]
                    self.data[split]["negative"] = \
                        [(i[1].lower().strip(), split_into_words[i[0]],
                         i[2].lower().strip(), int(i[3])) for i in tuples if not int(i[3])]
                elif self.opt.data.rel == "relation":
                    self.data[split]["total"] = \
                        [(i[1].lower().strip(), "<{}>".format(i[0]),
                         i[2].lower().strip(), int(i[3])) for i in tuples]
                    self.data[split]["positive"] = \
                        [(i[1].lower().strip(), "<{}>".format(i[0]),
                         i[2].lower().strip(), int(i[3])) for i in tuples if int(i[3])]
                    self.data[split]["negative"] = \
                        [(i[1].lower().strip(), "<{}>".format(i[0]),
                         i[2].lower().strip(), int(i[3])) for i in tuples if not int(i[3])]
            else:
                if self.opt.data.rel == "language":
                    self.data[split]["total"] = \
                        [(i[1].lower().strip(), split_into_words[i[0]],
                         i[2].lower().strip(), i[3]) for i in tuples]
                elif self.opt.data.rel == "relation":
                    self.data[split]["total"] = \
                        [(i[1].lower().strip(), "<{}>".format(i[0]),
                         i[2].lower().strip(), i[3]) for i in tuples]

        return False

    def make_tensors(self, text_encoder, special,
                     splits=["train", "dev", "test"], test=False):
        self.vocab_encoder = text_encoder.encoder
        self.vocab_decoder = text_encoder.decoder
        self.special_chars = special

        sequences = {}
        for split in splits:
            sequences[split], discarded = get_generation_sequences(
                self.data, split, text_encoder, test, self.opt.data.maxe1,
                self.opt.data.maxe2)

            if split == "train":
                self.data[split]["total"] = [j for i, j in enumerate(
                    self.data[split]["total"]) if i not in set(discarded)]
            self.masks[split]["total"] = [(len(i[0]), len(i[1]), len(i[2])) for
                                          i in sequences[split]]

        self.max_e1 = max([max([l[0] for l in self.masks[split]["total"]])
                           for split in self.masks])
        self.max_r = max([max([l[1] for l in self.masks[split]["total"]])
                          for split in self.masks])
        self.max_e2 = max([max([l[2] for l in self.masks[split]["total"]])
                           for split in self.masks])

        print(self.max_e1)
        print(self.max_r)
        print(self.max_e2)

        for split in splits:
            num_elements = len(sequences[split])
            self.sequences[split]["total"] = torch.LongTensor(
                num_elements, self.max_e1 + self.max_e2 + self.max_r).fill_(0)

            for i, seq in enumerate(sequences[split]):
                # print(self.sequences[split]["total"][i, :len(seq[0])].size())
                # print(torch.FloatTensor(seq[0]).size())
                self.sequences[split]["total"][i, :len(seq[0])] = \
                    torch.LongTensor(seq[0])
                start_r = self.max_e1
                end_r = self.max_e1 + len(seq[1])
                self.sequences[split]["total"][i, start_r:end_r] = \
                    torch.LongTensor(seq[1])
                start_e2 = self.max_e1 + self.max_r
                end_e2 = self.max_e1 + self.max_r + len(seq[2])
                self.sequences[split]["total"][i, start_e2:end_e2] = \
                    torch.LongTensor(seq[2])

            if split in ["test", "dev"]:
                print(split)
                self.sequences[split]["negative"] = \
                    self.sequences[split]["total"].index_select(
                        0, torch.LongTensor([i for i, j in enumerate(
                            self.data[split]['total']) if not j[3]]))
                            # self.data[split]['total'][:self.sequences[split]["total"].size(0)]) if not j[3]]))
                self.sequences[split]["positive"] = \
                    self.sequences[split]["total"].index_select(
                        0, torch.LongTensor([i for i, j in enumerate(
                            self.data[split]['total']) if j[3]]))
                            # self.data[split]['total'][:self.sequences[split]["total"].size(0)]) if j[3]]))

    def sample_batch(self, split, bs, cat="total", idxs=None):
        offset = self.offsets[split][cat]

        batch = {}

        # Decided not to reduce computation on here because it's all parallel
        # anyway and we don't want to run out of memory in cases where we
        # don't see the longest version quickly enough

        if idxs:
            seqs = self.sequences[split][cat].index_select(
                0, torch.LongTensor(idxs).to(
                    self.sequences[split][cat].device))
        else:
            seqs = self.sequences[split][cat][offset:offset + bs]
        batch["sequences"] = seqs.to(cfg.device)
        batch["attention_mask"] = make_attention_mask(seqs)
        batch["loss_mask"] = make_loss_mask(seqs, self.max_e1 + self.max_r)
        batch["key"] = (cat, offset, offset + bs)

        offset += seqs.size(0)

        self.offsets[split][cat] = offset

        if split == "train" and offset + bs > len(self.sequences[split][cat]):
            return batch, True
        elif offset >= len(self.sequences[split][cat]):
            return batch, True
        else:
            return batch, False

    def reset_offsets(self, splits=["train", "test", "dev"],
                      shuffle=True, keys=None):
        if isinstance(splits, str):
            splits = [splits]

        for split in splits:
            if keys is None:
                keys = ["total", "positive", "negative"]

            for key in keys:
                self.offsets[split][key] = 0

            if shuffle:
                self.shuffle_sequences(split, keys)

    def shuffle_sequences(self, split="train", keys=None):
        if keys is None:
            # print(type(self.data))
            # print(type(self.data.keys()))
            keys = self.data[split].keys()

        for key in keys:
            if key in ["positive", "negative"]:
                continue
            idxs = list(range(len(self.data[split][key])))

            random.shuffle(idxs)

            self.sequences[split][key] = \
                self.sequences[split][key].index_select(
                    0, torch.LongTensor(idxs))

            temp = [self.data[split][key][i] for i in idxs]
            self.data[split][key] = temp

            temp = [self.masks[split][key][i] for i in idxs]
            self.masks[split][key] = temp


def make_attention_mask(sequences):
    return (sequences != 0).float().to(cfg.device)


def make_loss_mask(sequences, max_event):
    # print(sequences.size())
    mask = (sequences != 0).float()
    mask[:, :max_event] = 0
    return mask[:, 1:].to(cfg.device)


def get_generation_sequences(data, split, text_encoder, test,
                             max_e1=10, max_e2=15):
    sequences = []
    count = 0

    final_event1 = None
    final_event2 = None
    final_relation = None

    discarded = []

    for event1, relation, event2, _ in tqdm(data[split]["total"]):
        e1, r, e2 = do_example(text_encoder, event1, relation, event2)

        if (split == "train" and len(e1) > max_e1 or
                len(e2) > max_e2):
            discarded.append(count)
            count += 1
            continue

        final = compile_final_sequence(
            e1, e2, r, text_encoder)

        sequences.append(final)

        count += 1

        if count > 10 and test:
            break

    return sequences, discarded


def do_example(text_encoder, event1, relation, event2):
    final_event1 = text_encoder.encode([event1], verbose=False)[0]
    if relation.lower() != relation:
        final_relation = [text_encoder.encoder[relation]]
    else:
        final_relation = text_encoder.encode(
            [relation], verbose=False)[0]
    if event2 is not None:
        final_event2 = text_encoder.encode([event2], verbose=False)[0]
    else:
        final_event2 = None

    return final_event1, final_relation, final_event2


def compile_final_sequence(final_event1, final_event2, final_relation, text_encoder):
    final = []

    final.append(final_event1)
    final.append(final_relation)
    final.append(final_event2)

    final[-1].append(text_encoder.encoder["<END>"])

    return final
