import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from networks.rank_cp import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *


def main(configs, fold_id):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    # if configs.split == 'split20':
    valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = Network(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.gnn.parameters()) + list(model.pred.parameters()) + list(model.rank.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]
    optimizer = AdamW(params, lr=configs.lr)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = (-1, -1, -1), (-1, -1, -1),(-1, -1, -1),None, None
    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = (-1, -1, -1),(-1, -1, -1),(-1, -1, -1), None, None
    early_stop_flag = None
    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch

            couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b, adj_b)
            loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
            loss_couple, _ = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b)
            loss = loss_couple + loss_e + loss_c
            loss = loss / configs.gradient_accumulation_steps

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        with torch.no_grad():
            model.eval()

            # if configs.split == 'split10':
            #     test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model)
            #     if test_ec[2] > metric_ec[2]:
            #         early_stop_flag = 1
            #         metric_ec, metric_e, metric_c = test_ec, test_e, test_c
            #     else:
            #         early_stop_flag += 1

            # if configs.split == 'split20':
            if configs.split == 'split10':
                valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c, _, _, _ = inference_one_epoch(configs, valid_loader, model)
                test_ec_p, test_ec_n, test_ec_avg, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model)
                if valid_ec_avg[2] > max_ec_avg[2]:
                    early_stop_flag = 1
                    max_ec_p, max_ec_n, max_ec_avg, max_e, max_c = valid_ec_p, valid_ec_n, valid_ec_avg, valid_e, valid_c
                    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = test_ec_p, test_ec_n, test_ec_avg, test_e, test_c
                else:
                    early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break
    return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c


def inference_one_batch(configs, batch, model):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch

    couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                      bert_clause_b, doc_len_b, adj_b)
    loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
    loss_couple, doc_couples_pred_b = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b, test=True)

    return to_np(loss_couple), to_np(loss_e), to_np(loss_c), \
           doc_couples_b, doc_couples_pred_b, doc_id_b, y_causes_b


def inference_one_epoch(configs, batches, model):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    y_causes_b_all = []
    for batch in batches:
        _, _, _, doc_couples, doc_couples_pred, doc_id_b, y_causes_b = inference_one_batch(configs, batch, model)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)
        y_causes_b_all.extend(list(y_causes_b))

    doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all)
    metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all, y_causes_b)
    return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all


def lexicon_based_extraction(doc_ids, couples_pred):
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 40
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 40
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    # for fold_id in range(1, n_folds+1):
    for fold_id in range(1, 2):
        print('===== fold {} ====='.format(fold_id))
        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp_pos: {}, P_ecp_pos: {}, R_ecp_pos: {}'.format(float_n(metric_ec_p[2]), float_n(metric_ec_p[0]), float_n(metric_ec_p[1])))
        print('F_ecp_neg: {}, P_ecp_neg: {}, R_ecp_neg: {}'.format(float_n(metric_ec_n[2]), float_n(metric_ec_n[0]), float_n(metric_ec_n[1])))
        print('F_ecp_avg: {}, P_ecp_avg: {}, R_ecp_avg: {}'.format(float_n(metric_ec_avg[2]), float_n(metric_ec_avg[0]), float_n(metric_ec_avg[1])))
        print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
        print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
        # print('F_ecp: {}'.format(float_n(metric_ec[2])))

        # metric_folds['ecp'].append(metric_ec)
        # metric_folds['emo'].append(metric_e)
        # metric_folds['cau'].append(metric_c)

    # metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    # metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    # metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    # print('===== Average =====')
    # print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    # print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    # print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    # write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c}, '{}_{}_metrics.pkl'.format(time.time(), configs.split))

