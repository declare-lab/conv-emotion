# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import time

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

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
    words = set(words)  
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) 
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
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def load_data(input_file, word_idx, max_doc_len, max_sen_len):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = [[] for i in range(7)]

    n_cut = 0
    inputFile = open(input_file, 'r')
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
        y_em, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            y_em[i][int(i+1 in emo)]=1
            y_ca[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        y_emotion.append(y_em)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    print('n_cut {}'.format(n_cut))
    return doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len

def get_y_pair_CR(doc_len, max_doc_len, y_pairs):
    y_pair = []
    for i in range(len(doc_len)):
        y_tmp = np.zeros((max_doc_len*max_doc_len, 2))
        for j in range(doc_len[i]):
            for k in range(doc_len[i]):
                if (j+1,k+1) in y_pairs[i]:
                    y_tmp[j*max_doc_len+k][1] = 1
                else :
                    y_tmp[j*max_doc_len+k][0] = 1
        y_pair.append(y_tmp)
    return y_pair

def get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs):
    y_pair, pair_cnt, pair_left_cnt = [], 0, 0
    for i in range(len(doc_len)):
        y_tmp = np.zeros((max_doc_len*(window_size*2+1), 2))
        for j in range(doc_len[i]):
            for k in range(-window_size,window_size+1):
                if (j+k) in range(doc_len[i]):
                    if (j+1,j+k+1) in y_pairs[i]:
                        y_tmp[j*(window_size*2+1)+k+window_size][1] = 1
                    else :
                        y_tmp[j*(window_size*2+1)+k+window_size][0] = 1
        y_pair.append(y_tmp)
        for j, k in y_pairs[i]:
            pair_cnt += 1
            if k-j not in range(-window_size,window_size+1):
                pair_left_cnt += 1
    print('pair_cnt {}, pair_left_cnt {}'.format(pair_cnt, pair_left_cnt))
    return y_pair, pair_left_cnt

def load_data_CR(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = load_data(input_file, word_idx, max_doc_len, max_sen_len)
    y_pair = get_y_pair_CR(doc_len, max_doc_len, y_pairs)
    
    y_emotion, y_cause, y_pair, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x, sen_len, doc_len

def load_data_WC(input_file, word_idx, max_doc_len = 75, max_sen_len = 45, window_size = 3):
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = load_data(input_file, word_idx, max_doc_len, max_sen_len)
    y_pair, pair_left_cnt = get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs)

    y_emotion, y_cause, y_pair, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x, sen_len, doc_len, pair_left_cnt

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

def _pair_prf_CR(pred_y, true_y, doc_len, nonneutral, threshold = 0.5):
    pred_num, acc_num, true_num = 0, 0, 0
    max_doc_len = int(np.sqrt(pred_y.shape[1]))
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            for k in range(doc_len[i]):
                idx = j*max_doc_len+k
                if nonneutral[i][j][1] == 1:
                    if pred_y[i][idx][1] > threshold:
                        pred_num += 1
                    if true_y[i][idx][1]>0.5:
                        true_num += 1
                    if true_y[i][idx][1]>0.5 and pred_y[i][idx][1] > threshold:
                        acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, nonneutral, threshold = 0.5):
    p_p, p_r, p_f = _pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, nonneutral, threshold = threshold)
    pred_y_pair = 1 - pred_y_pair
    true_y_pair = 1 - true_y_pair
    n_p, n_r, n_f = _pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, nonneutral, threshold = threshold)
    return p_f, n_f, (n_f + p_f) / 2

def pair_prf_WC(pred_y, true_y, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    p_p, p_r, p_f = _pair_prf_WC(pred_y, true_y, doc_len, pair_left_cnt, threshold, window_size)
    pred_y = 1 - pred_y
    true_y = 1 - true_y
    n_p, n_r, n_f = _pair_prf_WC(pred_y, true_y, doc_len, pair_left_cnt, threshold, window_size)
    return p_f, n_f, (n_f + p_f) / 2

def _pair_prf_WC(pred_y, true_y, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
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

def bert_word2id(words, max_sen_len_bert, tokenizer, i, x_tmp, sen_len_tmp):
    # 首先转换成unicode
    tokens_a, ret = tokenizer.tokenize(words), 0
    if len(tokens_a) > max_sen_len_bert - 2:
        ret += 1
        tokens_a = tokens_a[0:(max_sen_len_bert - 2)]
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    sen_len_tmp[i] = len(input_ids)
    for j in range(len(input_ids)):
        x_tmp[i][j] = input_ids[j]
    return ret

def load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len = [[] for i in range(9)]
    choice_len = []
    
    n_cut = 0
    inputFile = open(input_file, 'r')
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
        x_bert_tmp, sen_len_bert_tmp = np.zeros((max_doc_len, max_sen_len_bert),dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        x_tmp, sen_len_tmp = np.zeros((max_doc_len, max_sen_len),dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        choice_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
        for i in range(d_len):
            y_emotion_tmp[i][int(i+1 in emo)]=1
            y_cause_tmp[i][int(i+1 in cause)]=1
            text = inputFile.readline().strip().split(',')
            words = text[-1]
            n_cut += bert_word2id(words, max_sen_len_bert, tokenizer, i, x_bert_tmp, sen_len_bert_tmp)
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            choice_len_tmp[i] = min(int(text[0]), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    break
                x_tmp[i][j] = int(word_idx[word])
        
        y_emotion.append(y_emotion_tmp)
        y_cause.append(y_cause_tmp)
        x_bert.append(x_bert_tmp)
        sen_len_bert.append(sen_len_bert_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        choice_len.append(choice_len_tmp)
    print('n_cut {}'.format(n_cut))
    return doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len

def load_data_CR_Bert(input_file, tokenizer, word_idx, max_doc_len = 75, max_sen_len_bert = 60, max_sen_len = 30):
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len = load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len)
    y_pair = get_y_pair_CR(doc_len, max_doc_len, y_pairs)
    
    y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len = map(np.array, [y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x_bert', 'sen_len_bert', 'x', 'sen_len', 'doc_len', 'choice_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len

def load_data_WC_Bert(input_file, tokenizer, word_idx, max_doc_len = 75, max_sen_len_bert = 60, max_sen_len = 30, window_size = 3):
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len = load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len)
    y_pair, pair_left_cnt = get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs)
    
    y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len = map(np.array, [y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len, choice_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x_bert', 'sen_len_bert', 'x', 'sen_len', 'doc_len', 'choice_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, pair_left_cnt, choice_len
