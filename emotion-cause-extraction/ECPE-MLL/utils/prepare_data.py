# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

#用于生成minibatch训练数据
def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test: np.random.shuffle(index)
    for i in range(int( (length + batch_size -1) / batch_size ) ):
        ret = index[i * batch_size : (i + 1) * batch_size]
        if not test and len(ret) < batch_size : break
        yield ret
        
def load_w2v(embedding_dim, embedding_dim_pos, data_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile = open(data_file_path, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        d_len = int(line.strip().split()[1])
        inputFile.readline()
        for i in range(d_len):
            words.extend(inputFile.readline().strip().split(',')[-1].split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) 
    
    w2v = {}
    inputFile = open(embedding_path, 'r')
    inputFile.readline()
    for line in inputFile.readlines():
        line = line.strip().split()
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) 
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos



def bert_word2id_hier(words, tokenizer, i, x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp):
    # 首先转换成unicode
    tokens_a = tokenizer.tokenize(words)
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    start_idx = s_idx_bert_tmp[i]
    sen_len_tmp = len(input_ids)
    s_idx_bert_tmp[i+1] = start_idx + sen_len_tmp
    for j in range(sen_len_tmp):
        x_bert_tmp[start_idx+j] = input_ids[j]
        x_mask_bert_tmp[start_idx+j] = 1
        x_type_bert_tmp[start_idx+j] = i % 2

def cut_by_max_len(x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp, d_len, max_len=512):
    if s_idx_bert_tmp[d_len] > max_len:
        new_s_idx_bert_tmp = np.array(s_idx_bert_tmp)
        clause_max_len = max_len // d_len
        j = 0
        for i in range(d_len):
            start, end = s_idx_bert_tmp[i], s_idx_bert_tmp[i+1]
            if end-start <= clause_max_len:
                for k in range(start, end):
                    x_bert_tmp[j] = x_bert_tmp[k]
                    x_type_bert_tmp[j] = x_type_bert_tmp[k]
                    j+=1
                new_s_idx_bert_tmp[i+1] = new_s_idx_bert_tmp[i] + end - start
            else :
                for k in range(start, start+clause_max_len-1):
                    x_bert_tmp[j] = x_bert_tmp[k]
                    x_type_bert_tmp[j] = x_type_bert_tmp[k]
                    j+=1
                x_bert_tmp[j] = x_bert_tmp[end-1]
                x_type_bert_tmp[j] = x_type_bert_tmp[end-1]
                j+=1
                new_s_idx_bert_tmp[i+1] = new_s_idx_bert_tmp[i] + clause_max_len
        x_bert_tmp[j:] = 0
        x_mask_bert_tmp[j:] = 0
        x_type_bert_tmp[j:] = 0
        s_idx_bert_tmp = new_s_idx_bert_tmp
    x_bert_tmp = x_bert_tmp[:max_len]
    x_mask_bert_tmp = x_mask_bert_tmp[:max_len]
    x_type_bert_tmp = x_type_bert_tmp[:max_len]
    return x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp


def get_y_pair_rc(doc_len, max_doc_len, window_size, y_pairs):
    y_pair_r, y_pair_c, pair_cnt, pair_left_cnt = [], [], 0, 0
    for i in range(len(doc_len)):
        y_tmp_r = np.zeros((max_doc_len*(window_size*2+1), 2))
        y_tmp_c = np.zeros((max_doc_len*(window_size*2+1), 2))
        for j in range(doc_len[i]):
            for k in range(-window_size,window_size+1):
                if (j+k) in range(doc_len[i]):
                    if (j+1,j+k+1) in y_pairs[i]:
                        y_tmp_r[j*(window_size*2+1)+k+window_size][1] = 1
                    else :
                        y_tmp_r[j*(window_size*2+1)+k+window_size][0] = 1
                    if (j+k+1,j+1) in y_pairs[i]:
                        y_tmp_c[j*(window_size*2+1)+k+window_size][1] = 1
                    else :
                        y_tmp_c[j*(window_size*2+1)+k+window_size][0] = 1
        y_pair_r.append(y_tmp_r)
        y_pair_c.append(y_tmp_c)
        for j, k in y_pairs[i]:
            pair_cnt += 1
            if k-j not in range(-window_size,window_size+1):
                pair_left_cnt += 1
    print('pair_cnt {}, pair_left_cnt {}'.format(pair_cnt, pair_left_cnt))
    return y_pair_r, y_pair_c, pair_left_cnt

def load_data_bert_hier(input_file, tokenizer, word_idx, max_doc_len, max_sen_len, window_size = 3, max_doc_len_bert = 512):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len = [[] for i in range(11)]
    
    inputFile = open(input_file, 'r')
    cut_num = 0
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        emo, cause = zip(*pairs)
        y_emotion_tmp, y_cause_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2))
        # x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp = [np.zeros(1024, dtype=np.int32) for i in range(3)]
        x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp = [np.zeros(2000, dtype=np.int32) for i in range(3)]
        s_idx_bert_tmp = np.zeros(max_doc_len, dtype=np.int32)
        x_tmp, sen_len_tmp = np.zeros((max_doc_len, max_sen_len),dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        for i in range(d_len):
            y_emotion_tmp[i][int(i+1 in emo)]=1
            y_cause_tmp[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            bert_word2id_hier(words, tokenizer, i, x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp)
            # pdb.set_trace()
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    break
                x_tmp[i][j] = int(word_idx[word])
        cut_num = cut_num + int(s_idx_bert_tmp[d_len]>max_doc_len_bert)
        x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp = cut_by_max_len(x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp, d_len, max_len=max_doc_len_bert)

        y_emotion.append(y_emotion_tmp)
        y_cause.append(y_cause_tmp)
        x_bert.append(x_bert_tmp)
        x_mask_bert.append(x_mask_bert_tmp)
        x_type_bert.append(x_type_bert_tmp)
        s_idx_bert.append(s_idx_bert_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    print('\n\ncut_num: {}\n\n'.format(cut_num))
    y_pair_r, y_pair_c, pair_left_cnt = get_y_pair_rc(doc_len, max_doc_len, window_size, y_pairs)

    y_emotion, y_cause, y_pair_r, y_pair_c, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair_r, y_pair_c, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair_r', 'y_pair_c', 'x_bert', 'x_mask_bert', 'x_type_bert', 's_idx_bert', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair_r, y_pair_c, y_pairs, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, pair_left_cnt

def cal_prf(pred_y, true_y, doc_len, average='binary'): 
    pred_num, acc_num, true_num = 0, 0, 0
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            if pred_y[i][j]:
                pred_num += 1
            if true_y[i][j]:
                true_num += 1
            if pred_y[i][j] and true_y[i][j]:
                acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_pair_prf(pred_y, true_y, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_num, acc_num, true_num = 0, 0, pair_left_cnt
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]*(window_size*2+1)):
            if max(true_y[i][j]) > 1e-8:
                if pred_y[i][j][1] > threshold:
                    pred_num += 1
                if true_y[i][j][1]>0.5:
                    true_num += 1
                if true_y[i][j][1]>0.5 and pred_y[i][j][1] > threshold:
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_pair_prf_RCA(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_num, acc_num, true_num = 0, 0, pair_left_cnt
    wl, w = window_size*2+1, window_size
    for i in range(pred_y_R.shape[0]):
        for j in range(doc_len[i]*wl):
            if max(true_y_R[i][j]) > 1e-8:
                r,c = j//wl, j%wl-w
                c_idx = (r+c)*wl-c+w 
                if (pred_y_R[i][j][1]+pred_y_C[i][c_idx][1])/2 > threshold:
                    pred_num += 1
                if true_y_R[i][j][1]>0.5:
                    true_num += 1
                if true_y_R[i][j][1]>0.5 and (pred_y_R[i][j][1]+pred_y_C[i][c_idx][1])/2 > threshold:
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_pair_prf_RCAND(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_num, acc_num, true_num = 0, 0, pair_left_cnt
    wl, w = window_size*2+1, window_size
    for i in range(pred_y_R.shape[0]):
        for j in range(doc_len[i]*wl):
            if max(true_y_R[i][j]) > 1e-8:
                r,c = j//wl, j%wl-w
                c_idx = (r+c)*wl-c+w 
                if pred_y_R[i][j][1] > threshold and pred_y_C[i][c_idx][1] > threshold:
                    pred_num += 1
                if true_y_R[i][j][1]>0.5:
                    true_num += 1
                if true_y_R[i][j][1]>0.5 and (pred_y_R[i][j][1] > threshold and pred_y_C[i][c_idx][1] > threshold):
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_pair_prf_RCOR(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_num, acc_num, true_num = 0, 0, pair_left_cnt
    wl, w = window_size*2+1, window_size
    for i in range(pred_y_R.shape[0]):
        for j in range(doc_len[i]*wl):
            if max(true_y_R[i][j]) > 1e-8:
                r,c = j//wl, j%wl-w
                c_idx = (r+c)*wl-c+w 
                if pred_y_R[i][j][1] > threshold or pred_y_C[i][c_idx][1] > threshold:
                    pred_num += 1
                if true_y_R[i][j][1]>0.5:
                    true_num += 1
                if true_y_R[i][j][1]>0.5 and (pred_y_R[i][j][1] > threshold or pred_y_C[i][c_idx][1] > threshold):
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_pair_prf_RCAANDOR(pred_y_R, pred_y_C, true_y_R, true_y_C, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    ret = []
    p, r, f = cal_pair_prf(pred_y_R, true_y_R, doc_len, pair_left_cnt = pair_left_cnt, threshold = threshold, window_size = window_size)
    ret.extend([p, r, f])
    p, r, f = cal_pair_prf(pred_y_C, true_y_C, doc_len, pair_left_cnt = pair_left_cnt, threshold = threshold, window_size = window_size)
    ret.extend([p, r, f])
    p, r, f = cal_pair_prf_RCA(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = pair_left_cnt, threshold = threshold, window_size = window_size)
    ret.extend([p, r, f])
    p, r, f = cal_pair_prf_RCAND(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = pair_left_cnt, threshold = threshold, window_size = window_size)
    ret.extend([p, r, f])
    p, r, f = cal_pair_prf_RCOR(pred_y_R, pred_y_C, true_y_R, doc_len, pair_left_cnt = pair_left_cnt, threshold = threshold, window_size = window_size)
    ret.extend([p, r, f])
    return ret

def _cal_ecp_prf(eval_input, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_y_emo, true_y_emo, pred_y_cause, true_y_cause, pred_y_R, pred_y_C, true_y_R, true_y_C, doc_len = eval_input
    emo_cause_prf = []
    p, r, f = cal_prf(pred_y_emo, true_y_emo, doc_len)
    emo_cause_prf.extend([p, r, f])
    p, r, f = cal_prf(pred_y_cause, true_y_cause, doc_len)
    emo_cause_prf.extend([p, r, f])  
    pair_prf = cal_pair_prf_RCAANDOR(pred_y_R, pred_y_C, true_y_R, true_y_C, doc_len, pair_left_cnt, threshold, window_size)
    return emo_cause_prf, pair_prf

def cal_ecp_prf(eval_input, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    positive_emo_cause_prf, positive_pair_prf = _cal_ecp_prf(eval_input, pair_left_cnt, threshold, window_size)
    pred_y_emo, true_y_emo, pred_y_cause, true_y_cause, pred_y_R, pred_y_C, true_y_R, true_y_C, doc_len = eval_input
    pred_y_emo = 1 - pred_y_emo
    true_y_emo = 1 - true_y_emo
    pred_y_cause = 1 - pred_y_cause
    true_y_cause = 1 - true_y_cause
    pred_y_R, pred_y_C, true_y_R, true_y_C = 1-pred_y_R, 1-pred_y_C, 1-true_y_R, 1-true_y_C
    eval_input = (pred_y_emo, true_y_emo, pred_y_cause, true_y_cause, pred_y_R, pred_y_C, true_y_R, true_y_C, doc_len)
    negative_pair_prf, negative_pair_prf = _cal_ecp_prf(eval_input, pair_left_cnt, threshold, window_size)
    pair_prf = (np.array(positive_pair_prf) + np.array(negative_pair_prf))/2
    return positive_emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf
