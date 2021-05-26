import pickle, json, decimal, math
import numpy as np


def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))


def eval_func(doc_couples_all, doc_couples_pred_all, y_causes_b_all):
    tmp_num = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}
    tmp_den_p = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}
    tmp_den_r = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}

    for doc_couples, doc_couples_pred, y_causes_b in zip(doc_couples_all, doc_couples_pred_all, y_causes_b_all):
        doc_couples_total = []
        for i,o in enumerate(y_causes_b):
            if o == 1:
                doc_couples_total.extend([f'{i+1},{ii+1}' for ii in range(len(y_causes_b))])
        doc_couples = set([','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples])
        doc_couples_pred = set([','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples_pred])

        tmp_num['ec_p'] += len(doc_couples & doc_couples_pred)
        tmp_den_p['ec_p'] += len(doc_couples_pred)
        tmp_den_r['ec_p'] += len(doc_couples)

        doc_couples = set(doc_couples_total) - set(doc_couples)
        doc_couples_pred = set(doc_couples_total) - set(doc_couples_pred)
        tmp_num['ec_n'] += len(doc_couples & doc_couples_pred)
        tmp_den_p['ec_n'] += len(doc_couples_pred)
        tmp_den_r['ec_n'] += len(doc_couples)

        doc_emos = set([doc_couple.split(',')[0] for doc_couple in doc_couples])
        doc_emos_pred = set([doc_couple.split(',')[0] for doc_couple in doc_couples_pred])
        tmp_num['e'] += len(doc_emos & doc_emos_pred)
        tmp_den_p['e'] += len(doc_emos_pred)
        tmp_den_r['e'] += len(doc_emos)

        doc_caus = set([doc_couple.split(',')[1] for doc_couple in doc_couples])
        doc_caus_pred = set([doc_couple.split(',')[1] for doc_couple in doc_couples_pred])
        tmp_num['c'] += len(doc_caus & doc_caus_pred)
        tmp_den_p['c'] += len(doc_caus_pred)
        tmp_den_r['c'] += len(doc_caus)

    metrics = {}
    for task in ['ec_p','ec_n', 'e', 'c']:
        p = tmp_num[task] / (tmp_den_p[task] + 1e-8)
        r = tmp_num[task] / (tmp_den_r[task] + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        metrics[task] = (p, r, f)
    metrics['ec_avg'] = (np.array(metrics['ec_p']) + np.array(metrics['ec_n'])) / 2

    return metrics['ec_p'], metrics['ec_n'], metrics['ec_avg'], metrics['e'], metrics['c']


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js
