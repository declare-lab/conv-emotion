# encoding: utf-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
from sklearn.model_selection import KFold
import sys, os, time, pdb
sys.path.append('./bert')
sys.path.append('./utils')
from tf_funcs import *
from prepare_data import *
import modeling, optimization, tokenization


FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## input struct ##
# tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per clause')
# tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of clauses per document')
tf.app.flags.DEFINE_integer('max_sen_len', 50, 'max number of tokens per clause')
tf.app.flags.DEFINE_integer('max_doc_len', 125, 'max number of clauses per document')
tf.app.flags.DEFINE_integer('max_doc_len_bert', 350, 'max number of tokens per document for Bert Model')
## model struct ##
tf.app.flags.DEFINE_integer('model_iter_num', 1, 'iter num of ISML')
tf.app.flags.DEFINE_string('model_type', 'Inter-EC', 'model type: Inter-CE, Inter-EC, ISML, BERT')
tf.app.flags.DEFINE_integer('window_size', 3, 'window_size')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('start_fold', 1, 'start_fold') 
tf.app.flags.DEFINE_integer('end_fold', 11, 'end_fold') 
tf.app.flags.DEFINE_string('split', 'split10', 'split type: split10, split20')
tf.app.flags.DEFINE_string('bert_base_dir', './BERT/BERT-base-english/', 'base dir of pretrained bert')
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch size') 
tf.app.flags.DEFINE_float('learning_rate', 2e-5, 'learning rate') 
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'keep prob for word embedding')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'keep prob for softmax layer')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_float('emo', 1., 'loss weight of emotion ext.')
tf.app.flags.DEFINE_float('cause', 1., 'loss weight of cause ext.')
tf.app.flags.DEFINE_float('pair', 1., 'loss weight of pair ext.')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold for pair ext.')
tf.app.flags.DEFINE_integer('training_iter', 40, 'number of training iter') # for Bert
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')

    
def build_subtasks(x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training):
    def get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, is_training, feature_mask, scope='bert'):
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_base_dir + 'bert_config.json')
        bert_config.hidden_dropout_prob, bert_config.attention_probs_dropout_prob = 0.1, 0.2
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=x_bert,
            input_mask=x_mask_bert,
            token_type_ids=x_type_bert,
            scope=scope)
        s_bert, _ = model.get_sequence_output()
        # [-1, FLAGS.max_doc_len_bert, n_hidden]
        batch_size, n_hidden = tf.shape(s_bert)[0], s_bert.shape[-1].value
        index = tf.reshape(tf.range(0, batch_size) * FLAGS.max_doc_len_bert, [batch_size, 1]) + s_idx_bert
        index = tf.reshape(index, [-1])
        s_bert = tf.gather(tf.reshape(s_bert, [-1, n_hidden]), index)  # batch_size * n_hidden
        s_bert = tf.reshape(s_bert, [-1, FLAGS.max_doc_len, n_hidden])
        # [-1, FLAGS.max_doc_len, n_hidden]
        s_bert = s_bert * feature_mask # independent clause representation from bert 
        s_bert = tf.layers.dense(s_bert, 2 * FLAGS.n_hidden, use_bias=True)
        # shape: [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden]
        return s_bert

    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'):  
            inputs = biLSTM(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope='word_layer' + name)
        # inputs shape:        [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s
    
    def emo_cause_prediction(s_ec, is_training, name):
        s1 = tf.nn.dropout(s_ec, keep_prob = is_training * FLAGS.keep_prob2 + (1.-is_training))
        s1 = tf.reshape(s1, [-1, 2 * FLAGS.n_hidden])
        w_ec = get_weight_varible('softmax_w_'+name, [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_ec = get_weight_varible('softmax_b_'+name, [FLAGS.n_class])
        pred_ec = tf.nn.softmax(tf.matmul(s1, w_ec) + b_ec)
        pred_ec = tf.reshape(pred_ec, [-1, FLAGS.max_doc_len, FLAGS.n_class])
        return pred_ec, w_ec, b_ec
    
    cause_list, emo_list, reg = [], [], 0
    feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])

    if FLAGS.model_type in ['Inter-CE','Inter-EC']:
        with tf.name_scope('emotion_prediction'):
            s_ec = get_s(x, sen_len, name='word_encode_emotion')
            s_emo = biLSTM(s_ec, doc_len, n_hidden=FLAGS.n_hidden, scope='sentence_encode_emotion')
            pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')
        
        with tf.name_scope('cause_prediction'):
            s_ec = get_s(x, sen_len, name='word_encode_cause')
            s_ec = tf.concat([s_ec, pred_emo], axis=2) * feature_mask
            s_cause = biLSTM(s_ec, doc_len, n_hidden=FLAGS.n_hidden, scope='sentence_encode_cause')
            pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')

        emo_list.append(pred_emo)
        cause_list.append(pred_cause)
        reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
        reg += tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)

        if FLAGS.model_type in ['Inter-CE']:
            print('using Inter-CE')
            cause_list, emo_list, s_cause, s_emo = emo_list, cause_list, s_emo, s_cause
        else:
            print('using Inter-EC')
    elif FLAGS.model_type in ['ISML']:
        print('using ISML')
        for i in range(FLAGS.model_iter_num):
            s_ec = get_s(x, sen_len, name='word_encode_emotion_'+str(i))
            s_ec = tf.concat([s_ec] + cause_list + emo_list, axis=2) * feature_mask
            s_emo = biLSTM(s_ec, doc_len, n_hidden=FLAGS.n_hidden, scope='sentence_encode_emotion_' + str(i))
            pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion_'+str(i))

            s_ec = get_s(x, sen_len, name='word_encode_cause_'+str(i))
            s_ec = tf.concat([s_ec] + cause_list + emo_list, axis=2) * feature_mask
            s_cause = biLSTM(s_ec, doc_len, n_hidden=FLAGS.n_hidden, scope='sentence_encode_cause_' + str(i))
            pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause_'+str(i))

            emo_list.append(pred_emo)
            cause_list.append(pred_cause)
            reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
            reg += tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    else:
        print('using BERT')
        s_bert = get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, is_training, feature_mask, scope="bert_emotion")
    
        s_emo = standard_trans(s_bert, 2 * FLAGS.n_hidden, n_head=1, scope='sentence_encode_emo')
        pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

        s_cause = standard_trans(s_bert, 2 * FLAGS.n_hidden, n_head=1, scope='sentence_encode_cause')
        pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')

        emo_list.append(pred_emo)
        cause_list.append(pred_cause)
        reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
        reg += tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)


    return emo_list, cause_list, s_emo, s_cause, reg

def build_model(word_embedding, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training):
    x = tf.nn.embedding_lookup(word_embedding, x)
    x = tf.reshape(x, [-1, FLAGS.max_sen_len, 200])
    x = tf.nn.dropout(x, keep_prob = is_training * FLAGS.keep_prob1 + (1.-is_training))
    sen_len = tf.reshape(sen_len, [-1])

    ########################################## emotion & cause extraction  ############
    print('building subtasks')
    emo_list, cause_list, s_emo, s_cause, reg = build_subtasks(x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training)
    print('build subtasks Done!')

    ########################################## emotion-cause pair extraction  ############
    def pair_prediction(inputs, feature_num, scope="pair_prediction"):
        inputs = tf.reshape(inputs, [-1, feature_num])
        w_pair = get_weight_varible(scope+'_softmax_w_pair', [feature_num, FLAGS.n_class * (FLAGS.window_size*2+1)])
        b_pair = get_weight_varible(scope+'_softmax_b_pair', [FLAGS.n_class * (FLAGS.window_size*2+1)])
        s_rc = tf.matmul(inputs, w_pair) + b_pair
        pred_pair = tf.nn.softmax(tf.reshape(s_rc, [-1, 2]))
        pred_pair = tf.reshape(pred_pair, [-1, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
        reg_tmp = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
        return pred_pair, reg_tmp

    pred_pair_row, reg_tmp = pair_prediction(s_emo, FLAGS.n_hidden * 2, scope="pair_row_prediction")
    reg += reg_tmp

    pred_pair_col, reg_tmp = pair_prediction(s_cause, FLAGS.n_hidden * 2, scope="pair_col_prediction")
    reg += reg_tmp
        
    return emo_list, cause_list, pred_pair_row, pred_pair_col, reg

def print_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('model_type {} \nwindow_size {} \nbert_base_dir {} \nmodel_iter_num {}'.format(
        FLAGS.model_type, FLAGS.window_size, FLAGS.bert_base_dir, FLAGS.model_iter_num))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('batch {} \nlr {} \nkb1 {} \nkb2 {} \nl2_reg {}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('FLAGS.emo {} \nFLAGS.cause {} \nFLAGS.pair {} \nthreshold {} \ntraining_iter {}\n\n'.format(
        FLAGS.emo,  FLAGS.cause, FLAGS.pair, FLAGS.threshold, FLAGS.training_iter))

class Dataset(object):
    def __init__(self, data_file_name, tokenizer, word_id_mapping):
        doc_id, y_emotion, y_cause, y_pair_row, y_pair_col, y_pairs, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, pair_left_cnt = load_data_bert_hier(data_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len, FLAGS.window_size, FLAGS.max_doc_len_bert)
        self.doc_id, self.y_pairs = doc_id, y_pairs
        self.y_emotion, self.y_cause, self.y_pair_row, self.y_pair_col,  = y_emotion, y_cause, y_pair_row, y_pair_col
        self.x_bert, self.x_mask_bert, self.x_type_bert, self.s_idx_bert = x_bert, x_mask_bert, x_type_bert, s_idx_bert
        self.x, self.sen_len, self.doc_len = x, sen_len, doc_len
        self.pair_left_cnt = pair_left_cnt

def get_batch_data(dataset, is_training, batch_size, test=False):
    ds = dataset
    for index in batch_index(len(ds.y_cause), batch_size, test):
        feed_list = [ds.x_bert[index], ds.x_mask_bert[index], ds.x_type_bert[index], ds.s_idx_bert[index], ds.x[index], ds.sen_len[index], ds.doc_len[index], is_training, ds.y_emotion[index], ds.y_cause[index], ds.y_pair_row[index], ds.y_pair_col[index]]
        yield feed_list, len(index)

def run():
    # if FLAGS.log_file_name:
    #     if not os.path.exists('log'):
    #         os.makedirs('log')
    #     sys.stdout = open('log/'+FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, _ = load_w2v(embedding_dim=200, embedding_dim_pos=50, data_file_path='data_reccon/all_data_pair.txt', embedding_path='data_reccon/glove.6B.200d.txt')
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')

    print('build model...')
    x_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert]) # for Bert
    x_mask_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert]) # for Bert
    x_type_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert]) # for Bert
    s_idx_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len]) # for Bert
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.float32) 
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_pair_row = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    y_pair_col = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    placeholders = [x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training, y_emotion, y_cause, y_pair_row, y_pair_col]  
    
    
    emo_list, cause_list, pred_pair_row, pred_pair_col, reg = build_model(word_embedding, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training)
    print('build model done!\n')

    loss_emo, loss_cause = [0.]*2
    for i in range(len(emo_list)):
        loss_emo += - tf.reduce_sum(y_emotion * tf.log(emo_list[i])) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
        loss_cause += - tf.reduce_sum(y_cause * tf.log(cause_list[i])) / tf.cast(tf.reduce_sum(y_cause), dtype=tf.float32)
    loss_pair = - tf.reduce_sum(y_pair_row * tf.log(pred_pair_row)) / tf.cast(tf.reduce_sum(y_pair_row), dtype=tf.float32)
    loss_pair += - tf.reduce_sum(y_pair_col * tf.log(pred_pair_col)) / tf.cast(tf.reduce_sum(y_pair_col), dtype=tf.float32)
    
    loss_op = loss_emo * FLAGS.emo + loss_cause * FLAGS.cause + loss_pair * FLAGS.pair + reg * FLAGS.l2_reg
    
    def get_bert_optimizer(loss_op):
        num_train_steps = int(1750 / FLAGS.batch_size * FLAGS.training_iter)
        num_warmup_steps = int(num_train_steps * 0.1)
        print('\n\nnum_warmup_steps {}\n\n'.format(num_warmup_steps))
        optimizer, run_lr = optimization.create_bert_optimizer(loss_op, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)
        return optimizer, run_lr

    def init_from_bert_checkpoint():
        init_checkpoint = FLAGS.bert_base_dir + 'bert_model.ckpt'
        tvars = tf.trainable_variables()
        (assignment_map_emotion, initialized_variable_names_emotion) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, bert_scope='bert_emotion')
        (assignment_map_cause, initialized_variable_names_cause) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, bert_scope='bert_cause')
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map_emotion) 
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map_cause) 
        # 仅仅只是替换变量的 initializers，后面运行sess.run(tf.global_variables_initializer())时才真正的加载
        
        tf.logging.info("**** Trainable Variables ****")
        initialized_variable_names_emotion.update(initialized_variable_names_cause)
        initialized_variable_names = initialized_variable_names_emotion
        for idx, var in enumerate(tvars):
            init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""
            tf.logging.info("var-index %s:  name = %s, shape = %s%s", idx, var.name, var.shape, init_string)

    def get_bert_tokenizer():
        do_lower_case = True
        tokenization.validate_case_matches_checkpoint(do_lower_case, FLAGS.bert_base_dir + 'bert_model.ckpt')
        return tokenization.FullTokenizer(vocab_file = FLAGS.bert_base_dir + 'vocab.txt', do_lower_case = do_lower_case)

    
    if FLAGS.model_type=='BERT':
        optimizer, run_lr_op = get_bert_optimizer(loss_op)
    else:
        optimizer, run_lr_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op), tf.constant(FLAGS.learning_rate)

    pred_emo, pred_cause = emo_list[-1], cause_list[-1]
    
    true_y_emo_op = tf.argmax(y_emotion, 2)
    pred_y_emo_op = tf.argmax(pred_emo, 2)
    true_y_cause_op = tf.argmax(y_cause, 2)
    pred_y_cause_op = tf.argmax(pred_cause, 2)

    # Training Code Block
    print_info()
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        de_emo_cause_list, de_pair_list = [], []
        te_emo_cause_list, te_pair_list = [], []
        tokenizer = get_bert_tokenizer()
        fold = FLAGS.start_fold
        
        # while True:
        # train for one fold
        # if fold == FLAGS.end_fold: 
        #     break
        print('############# fold {} begin ###############'.format(fold))
        init_from_bert_checkpoint()
        sess.run(tf.global_variables_initializer())

        # Data Code Block
        train_file_name = 'data_reccon/dailydialog_train.txt'
        dev_file_name = 'data_reccon/dailydialog_valid.txt'
        test_file_name = 'data_reccon/dailydialog_test.txt'
        # test_file_name = 'data_reccon/iemocap_test.txt'
        train_data = Dataset(train_file_name, tokenizer, word_id_mapping)
        test_data = Dataset(test_file_name, tokenizer, word_id_mapping)
        if FLAGS.split=='split20':
            dev_data = Dataset(dev_file_name, tokenizer, word_id_mapping)
        else:
            dev_data = test_data
        print('train docs: {}  valid docs: {}  test docs: {}'.format(len(train_data.x), len(dev_data.x), len(test_data.x)))
        

        de_emo_cause_max_prf, de_pair_max_prf = [-1.] * 6, [-1.]*15
        te_emo_cause_max_prf, te_pair_max_prf = [-1.] * 6, [-1.]*15
        tr_row_pair_max_f1 = -1.
        for i in range(FLAGS.training_iter):
            print(f'######################## epoch {i} ##########################')
            start_time, step = time.time(), 1
            batch_eval_inputs = []
            def combine_eval_inputs(x):
                ret = []
                for a in list(x):
                    ret.extend(list(a))
                return np.array(ret) 
            def print_result(emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf):
                p, r, f1 = emo_cause_prf[:3]
                print('emotion_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = emo_cause_prf[3:6]
                print('cause_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = positive_pair_prf[:3]
                print('POSITIVE: row pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = positive_pair_prf[3:6]
                print('POSITIVE: col pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = negative_pair_prf[:3]
                print('NEGATIVE: row pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = negative_pair_prf[3:6]
                print('NEGATIVE: col pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = pair_prf[:3]
                print('AVERAGE: row pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = pair_prf[3:6]
                print('AVERAGE: col pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = pair_prf[6:9]
                print('AVERAGE: row,col average pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = pair_prf[9:12]
                print('AVERAGE: row and col pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                p, r, f1 = pair_prf[12:15]
                print('AVERAGE: row or col pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))

            # train
            eval_input_op = [pred_y_emo_op, true_y_emo_op, pred_y_cause_op, true_y_cause_op, pred_pair_row, pred_pair_col, y_pair_row, y_pair_col, doc_len]
            for train, _ in get_batch_data(train_data, 1, FLAGS.batch_size):
                _, run_lr, loss = sess.run([optimizer, run_lr_op, loss_op], feed_dict=dict(zip(placeholders, train)))
                eval_input = sess.run(eval_input_op, feed_dict=dict(zip(placeholders, train)))
                batch_eval_inputs.append(eval_input)
                # if step % 10 == 0:
                #     print('TRAIN SET')
                #     print('step {}: train loss {:.4f} run_lr {:.6f}'.format(step, loss, run_lr))
                #     emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf = cal_ecp_prf(eval_input, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                #     print_result(emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf)
                #     # print('\ncost time: {:.1f}s'.format(time.time()-start_time))
                #     start_time = time.time()
                # if step % 60 == 0:
                #     batch_eval_inputs = zip(*batch_eval_inputs)
                #     eval_input = map(combine_eval_inputs, batch_eval_inputs)
                #     batch_eval_inputs = []
                #     emo_cause_prf, pair_prf = cal_ecp_prf(eval_input, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                #     print('\n############## Evaluation on {}-{} train steps ##############'.format(step-60,step))
                #     print_result(emo_cause_prf, pair_prf)
                #     if pair_prf[2] > tr_row_pair_max_f1:
                #         tr_row_pair_max_f1 = pair_prf[2]
                step = step + 1
            # test
            def evaluation(dataset):
                ds = dataset
                batch_eval_inputs = []
                for test, _ in get_batch_data(ds, 0., FLAGS.batch_size, test=True):
                    eval_input = sess.run(eval_input_op, feed_dict=dict(zip(placeholders, test)))
                    batch_eval_inputs.append(eval_input)

                batch_eval_inputs = zip(*batch_eval_inputs)
                eval_input = list(map(combine_eval_inputs, batch_eval_inputs))
                emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf = cal_ecp_prf(eval_input, ds.pair_left_cnt, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                return emo_cause_prf, positive_pair_prf, negative_pair_prf, pair_prf

            de_emo_cause_prf, de_positive_pair_prf, de_negative_pair_prf, de_pair_prf = evaluation(dev_data)
            te_emo_cause_prf, te_positive_pair_prf, te_negative_pair_prf, te_pair_prf = evaluation(test_data)

            for j in range(2,6,3):
                if de_emo_cause_prf[j] > de_emo_cause_max_prf[j]:
                    de_emo_cause_max_prf[j-2:j+1] = de_emo_cause_prf[j-2:j+1]
                    te_emo_cause_max_prf[j-2:j+1] = te_emo_cause_prf[j-2:j+1]

            for j in range(2,15,3):
                if de_pair_prf[j] > de_pair_max_prf[j]:
                    de_pair_max_prf[j-2:j+1] = de_pair_prf[j-2:j+1]
                    te_pair_max_prf[j-2:j+1] = te_pair_prf[j-2:j+1]


            print('DEV SET:')
            print_result(de_emo_cause_prf, de_positive_pair_prf, de_negative_pair_prf, de_pair_prf)
            print('TEST SET:')
            print_result(te_emo_cause_prf, te_positive_pair_prf, te_negative_pair_prf, te_pair_prf)
            print()
            # p, r, f1 = de_emo_cause_prf[:3]
            # print('Dev emotion_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
            # p, r, f1 = de_emo_cause_max_prf[:3]
            # print('Dev emotion_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(p, r, f1))
            # p, r, f1 = te_emo_cause_max_prf[:3]
            # print('Test emotion_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(p, r, f1))

            # p, r, f1 = de_emo_cause_prf[3:6]
            # print('Dev cause_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
            # p, r, f1 = de_emo_cause_max_prf[3:6]
            # print('Dev cause_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(p, r, f1))
            # p, r, f1 = te_emo_cause_max_prf[3:6]
            # print('Test cause_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(p, r, f1))

            # for j in [0,1,2,4]:
            #     tmp = ['row', 'col', 'avg', 'AND', 'OR']
            #     idx = 2+j*3
            #     p, r, f1 = de_pair_prf[idx-2:idx+1]
            #     print('\nDev {}_pair_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(tmp[j], p, r, f1 ))
            #     p, r, f1 = de_pair_max_prf[idx-2:idx+1]
            #     print('\nDev {}_pair_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(tmp[j], p, r, f1 ))
            #     p, r, f1 = te_pair_max_prf[idx-2:idx+1]
            #     print('\nTest {}_pair_prediction: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(tmp[j], p, r, f1 ))

        print('Optimization Finished!\n')
        # print('############# fold {} end ###############'.format(fold))
        
        if FLAGS.model_type != 'BERT' or tr_row_pair_max_f1 > 0.5:
            de_emo_cause_list.append(de_emo_cause_max_prf)
            de_pair_list.append(de_pair_max_prf)
            te_emo_cause_list.append(te_emo_cause_max_prf)
            te_pair_list.append(te_pair_max_prf)
            fold = fold + 1
        
              
        de_emo_cause_list, de_pair_list, te_emo_cause_list, te_pair_list = map(lambda x: np.array(x), [de_emo_cause_list, de_pair_list, te_emo_cause_list, te_pair_list])

        
        def finnal_output(emo_cause_list, pair_list):
            for i in range(5):
                tmp = ['row', 'col', 'avg', 'AND', 'OR']
                idx = 2+i*3
                print('\n{} pair_predict: test f1 in 10 fold: {}'.format(tmp[i], pair_list[:,idx:idx+1]))
                p, r, f1 = pair_list[:,idx-2:idx+1].mean(axis=0)
                print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

            print('\nemotion_prediction: test f1 in 10 fold: {}'.format(emo_cause_list[:,2:3]))
            p, r, f1 = emo_cause_list[:,:3].mean(axis=0)
            print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

            print('\ncause_prediction: test f1 in 10 fold: {}'.format(emo_cause_list[:,5:6]))
            p, r, f1 = emo_cause_list[:,3:6].mean(axis=0)
            print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        # print('############# Evaluation on Dev ############# ')
        # finnal_output(de_emo_cause_list, de_pair_list)
        # print('############# Evaluation on Test ############# ')
        # finnal_output(te_emo_cause_list, te_pair_list)
        # print_time()
     

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO) # for Bert

    ########################## Split 10 ##########################
    FLAGS.split='split10'
    
    FLAGS.training_iter, FLAGS.batch_size, FLAGS.learning_rate = 40, 32, 0.005
    FLAGS.model_type = 'Inter-EC'
    # FLAGS.log_file_name = '{}_Inter-EC_1.log'.format(FLAGS.split)
    # run()
    # FLAGS.log_file_name = '{}_Inter-EC_2.log'.format(FLAGS.split)
    # run()

    FLAGS.model_type = 'Inter-CE'
    # FLAGS.log_file_name = '{}_Inter-CE_1.log'.format(FLAGS.split)
    # run()
    # FLAGS.log_file_name = '{}_Inter-CE_2.log'.format(FLAGS.split)
    # run()

    FLAGS.model_type = 'ISML'
    FLAGS.model_iter_num = 1
    # FLAGS.log_file_name = '{}_Indep_1.log'.format(FLAGS.split)
    # run()
    # FLAGS.log_file_name = '{}_Indep_2.log'.format(FLAGS.split)
    # run()
    
    FLAGS.model_type = 'ISML'
    # for FLAGS.model_iter_num in range(2,8):
        # FLAGS.log_file_name = '{}_ISML{}_1.log'.format(FLAGS.split, FLAGS.model_iter_num)
        # run()
        # FLAGS.log_file_name = '{}_ISML{}_2.log'.format(FLAGS.split, FLAGS.model_iter_num)
        # run()

    FLAGS.training_iter, FLAGS.batch_size, FLAGS.learning_rate = 20, 4, 1e-5
    FLAGS.model_type = 'BERT'
    # FLAGS.log_file_name = '{}_BERT_1.log'.format(FLAGS.split)
    # run()
    # FLAGS.log_file_name = '{}_BERT_2.log'.format(FLAGS.split)
    # run()

    ########################## Split 20 ##########################
    FLAGS.split='split20'
    FLAGS.start_fold, FLAGS.end_fold = 1, 21
    
    FLAGS.training_iter, FLAGS.batch_size, FLAGS.learning_rate = 20, 32, 0.005
    FLAGS.model_type = 'ISML'
    FLAGS.model_iter_num = 6
    FLAGS.log_file_name = '{}_ISML{}.log'.format(FLAGS.split, FLAGS.model_iter_num)
    run()

    FLAGS.training_iter, FLAGS.batch_size, FLAGS.learning_rate = 20, 3, 2e-5
    FLAGS.model_type = 'BERT'
    FLAGS.log_file_name = '{}_BERT.log'.format(FLAGS.split)
    run()


if __name__ == '__main__':
    tf.app.run() 
