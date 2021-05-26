import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
from sklearn.model_selection import KFold
import sys, os, time, codecs

sys.path.append('./utils')
sys.path.append('./bert')
from tf_funcs import *
from prepare_data import *
import modeling, optimization, tokenization

FLAGS = tf.app.flags.FLAGS
tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data_reccon/glove.6B.200d.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
# tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of sentences per document')
tf.app.flags.DEFINE_integer('max_doc_len', 125, 'max number of sentences per document')
tf.app.flags.DEFINE_integer('max_sen_len_bert', 60, 'max number of tokens per sentence for Bert')
tf.app.flags.DEFINE_bool('choice_len', False, 'without future utterances')
## model struct ##
tf.app.flags.DEFINE_string('model_type', 'Inter-EC', 'model type: Indep, Inter-CE, Inter-EC')
tf.app.flags.DEFINE_string('trans_type', 'cross_road', 'transformer type: cross_road, window_constrained')
tf.app.flags.DEFINE_integer('window_size', 3, 'window_size')
tf.app.flags.DEFINE_integer('trans_iter', 1, 'number of cross-road 2D transformer layers')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
tf.app.flags.DEFINE_string('bert_base_dir', './BERT/BERT-base-english/', 'base dir of pretrained bert')
tf.app.flags.DEFINE_float('bert_hidden_kb', 0.9, 'keep prob for bert')
tf.app.flags.DEFINE_float('bert_attention_kb', 0.8, 'keep prob for bert')
tf.app.flags.DEFINE_string('scope', 'TEMP', 'scope')
tf.app.flags.DEFINE_integer('batch_size', 3, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 2e-5, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'keep prob for word embedding')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'keep prob for softmax layer')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_float('emo', 1., 'loss weight of emotion ext.')
tf.app.flags.DEFINE_float('cause', 1., 'loss weight of cause ext.')
tf.app.flags.DEFINE_float('pair', 1., 'loss weight of pair ext.')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold for pair ext.')
tf.app.flags.DEFINE_integer('feature_num', 30, 'feature vector length of pairs')
tf.app.flags.DEFINE_integer('training_iter', 40, 'number of training iter')

def build_subtasks(x_bert, sen_len_bert, x, sen_len, doc_len, is_training):
    def get_bert_s(x_bert, sen_len_bert, is_training, feature_mask):
        x_bert = tf.reshape(x_bert, [-1, FLAGS.max_sen_len_bert])
        sen_len_bert = tf.reshape(sen_len_bert, [-1])
        x_bert_mask = tf.cast(tf.sequence_mask(sen_len_bert, FLAGS.max_sen_len_bert), tf.int32)
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_base_dir + 'bert_config.json')
        bert_config.hidden_dropout_prob, bert_config.attention_probs_dropout_prob = 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=x_bert,
            input_mask=x_bert_mask)
        s_bert = model.get_pooled_output()

        s_bert = tf.reshape(s_bert, [-1, FLAGS.max_doc_len, s_bert.shape[-1].value])
        s_bert = s_bert * feature_mask # independent clause representation from bert 
        s_bert = tf.layers.dense(s_bert, 2 * FLAGS.n_hidden, use_bias=True)
        # shape: [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden]
        return s_bert
    
    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'):  
            inputs = biLSTM(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
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

    
    with tf.name_scope('emotion_prediction'):
        feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
        s1 = get_bert_s(x_bert, sen_len_bert, is_training, feature_mask)
        s_emo = standard_trans(s1, 2 * FLAGS.n_hidden, n_head=1, scope=FLAGS.scope + 'sentence_encode_emo_')
        pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

    with tf.name_scope('cause_prediction'):
        if True:
            s1 = get_s(x, sen_len, name='word_encode_cause')
            s1 = tf.concat([s1, tf.stop_gradient(pred_emo)], 2) * feature_mask
            s_cause = biLSTM(s1, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_cause')
            pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')
        else:
            # s1 = get_s(x, sen_len, name='word_encode_cause')
            s2 = tf.concat([s1, tf.stop_gradient(pred_emo)], 2) * feature_mask
            s_cause = biLSTM(s2, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_cause')
            pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')


    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
    return pred_emo, pred_cause, tf.stop_gradient(s_emo), tf.stop_gradient(s_cause), reg

def pair_prediction(inputs, feature_num, scope="pair_prediction"):
    inputs = tf.reshape(inputs, [-1, feature_num])
    w_pair = get_weight_varible(scope+'_softmax_w_pair', [feature_num, FLAGS.n_class])
    b_pair = get_weight_varible(scope+'_softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(inputs, w_pair) + b_pair)
    reg_tmp = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
    return pred_pair, reg_tmp

def build_maintask_WC(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding):
    ####################################### pair features ################################################
    print('pair features')
    batch = tf.shape(s_emo)[0]
    conc0 = tf.zeros([batch, 2 * FLAGS.n_hidden])
    pair_x = []
    for i in range(FLAGS.max_doc_len):
        for j in range(i-FLAGS.window_size,i+FLAGS.window_size+1):
            conc_i = s_emo[:,i,:]
            conc_j = s_cause[:,j,:] if j in range(FLAGS.max_doc_len) else conc0
            pred_emo_feature_i = pred_emo_feature[:,i,:]
            pred_cause_feature_j = pred_cause_feature[:,j,:] if j in range(FLAGS.max_doc_len) else conc0[:,:2]
            relative_pos = tf.nn.embedding_lookup(pos_embedding, tf.ones([batch], tf.int32) * (j-i+100) )
            ns = tf.concat([conc_i, conc_j, pred_emo_feature_i, pred_cause_feature_j, relative_pos], 1)
            pair_x.append(ns)
    pair_x = tf.transpose(tf.cast(pair_x, tf.float32), perm=[1, 0, 2])
    pair_x = tf.layers.dense(pair_x, FLAGS.feature_num, use_bias=True, activation=tf.nn.relu)
    print('pair features Done!')

    ########################### pair interaction & prediction ########################################################################
    print('pair interaction')
    for i in range(FLAGS.trans_iter):
        pair_x = standard_trans(pair_x, n_hidden = FLAGS.feature_num, n_head = 1, scope="standard_trans{}".format(i))
    print('pair interaction Done!')

    pred_pair, reg_tmp = pair_prediction(pair_x, FLAGS.feature_num, scope="pair_prediction")
    pred_pair = tf.reshape(pred_pair, [-1, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    return pred_pair, reg_tmp

def build_maintask_CR(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding, doc_len, choice_len):
    ####################################### pair features ################################################
    print('pair features')
    feature_num = FLAGS.feature_num
    s_emo = tf.layers.dense(s_emo, feature_num, use_bias=True)
    s_cause = tf.layers.dense(s_cause, feature_num, use_bias=True)
    pred_emo_feature = tf.layers.dense(pred_emo_feature, feature_num, use_bias=True)
    pred_cause_feature = tf.layers.dense(pred_cause_feature, feature_num, use_bias=True)
    pos_embedding = tf.layers.dense(pos_embedding, feature_num, use_bias=True)
    ## 
    s_emo = tf.tile(tf.reshape(s_emo, [-1, FLAGS.max_doc_len, 1, feature_num]), [1,1,FLAGS.max_doc_len,1])
    s_cause = tf.tile(tf.reshape(s_cause, [-1, 1, FLAGS.max_doc_len, feature_num]), [1,FLAGS.max_doc_len,1,1])
    pred_emo_feature = tf.tile(tf.reshape(pred_emo_feature, [-1, FLAGS.max_doc_len, 1, feature_num]), [1,1,FLAGS.max_doc_len,1])
    pred_cause_feature = tf.tile(tf.reshape(pred_cause_feature, [-1, 1, FLAGS.max_doc_len, feature_num]), [1,FLAGS.max_doc_len,1,1])
    ##
    tmp = tf.cast(range(FLAGS.max_doc_len), tf.int32)
    abs_cause = tf.tile(tf.reshape(tmp, [1, FLAGS.max_doc_len]), [FLAGS.max_doc_len, 1])
    abs_emo = tf.tile(tf.reshape(tmp, [FLAGS.max_doc_len, 1]), [1, FLAGS.max_doc_len])
    relative_pos = tf.nn.embedding_lookup(pos_embedding, abs_cause - abs_emo + 100)
    relative_pos = tf.tile(tf.reshape(relative_pos, [1, FLAGS.max_doc_len, FLAGS.max_doc_len, feature_num]), [tf.shape(s_emo)[0],1,1,1])
    ##
    pair_x = tf.nn.relu(s_emo + s_cause + pred_emo_feature + pred_cause_feature + relative_pos)
    if FLAGS.choice_len:
        mask = tf.cast(tf.sequence_mask(choice_len, FLAGS.max_doc_len), tf.float32)
        mask = tf.expand_dims(mask, 3)
    else:
        mask = tf.cast(tf.sequence_mask(doc_len, FLAGS.max_doc_len), tf.float32)
        mask = tf.expand_dims(tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2), 3)
    pair_x = pair_x * mask
    # [batch, FLAGS.max_doc_len, FLAGS.max_doc_len, feature_num])
    print('pair features Done!')

    ########################### pair interaction & prediction ########################################################################
    print('pair interaction')
    for i in range(FLAGS.trans_iter):
        pair_x = CR_2Dtrans(pair_x, n_hidden = feature_num, n_head = 1, scope="CR_2Dtrans{}".format(i))
    print('pair interaction Done!')

    pred_pair, reg_tmp = pair_prediction(pair_x, feature_num, scope="pair_prediction")
    pred_pair = tf.reshape(pred_pair, [-1, FLAGS.max_doc_len * FLAGS.max_doc_len, FLAGS.n_class])
    return pred_pair, reg_tmp

def build_model(word_embedding, pos_embedding, x_bert, sen_len_bert, x, sen_len, doc_len,  is_training, choice_len,):
    x = tf.nn.embedding_lookup(word_embedding, x)
    x = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    x = tf.nn.dropout(x, keep_prob = is_training * FLAGS.keep_prob1 + (1.-is_training))
    sen_len = tf.reshape(sen_len, [-1])
    # x shape:        [-1, FLAGS.max_sen_len, FLAGS.embedding_dim]

    ########################################## emotion & cause extraction  ############
    print('building subtasks')
    pred_emo, pred_cause, s_emo, s_cause, reg = build_subtasks(x_bert, sen_len_bert, x, sen_len, doc_len, is_training)
    print('build subtasks Done!')
    feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
    pred_emo_feature = tf.stop_gradient(pred_emo * feature_mask + 1e-8)
    pred_cause_feature = tf.stop_gradient(pred_cause * feature_mask + 1e-8)

    ########################################## emotion-cause pair extraction  ############
    if FLAGS.trans_type=='cross_road':
        pred_pair, reg_tmp = build_maintask_CR(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding, doc_len, choice_len)
    else:
        pred_pair, reg_tmp = build_maintask_WC(s_emo, s_cause, pred_emo_feature, pred_cause_feature, pos_embedding)
    reg += reg_tmp
        
    return pred_emo, pred_cause, pred_pair, reg

def print_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('model_type {} \ntrans_type {} \ntrans_iter {} \nwindow_size {} \n'.format(
        FLAGS.model_type,  FLAGS.trans_type, FLAGS.trans_iter, FLAGS.window_size))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('batch {} \nlr {} \nkb1 {} \nkb2 {} \nl2_reg {}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('bert_base_dir {} \nbert_hidden_kb {} \nbert_attention_kb {}'.format(
        FLAGS.bert_base_dir, FLAGS.bert_hidden_kb, FLAGS.bert_attention_kb))
    print('FLAGS.emo {} \nFLAGS.cause {} \nFLAGS.pair {} \nthreshold {} \ntraining_iter {}\n\n'.format(
        FLAGS.emo,  FLAGS.cause, FLAGS.pair, FLAGS.threshold, FLAGS.training_iter))

def get_batch_data(x_bert, sen_len_bert, x, sen_len, doc_len, is_training, y_emotion, y_cause, y_pair, batch_size, choice_len, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x_bert[index], sen_len_bert[index], x[index], sen_len[index], doc_len[index], is_training, y_emotion[index], y_cause[index], y_pair[index], choice_len[index]]
        yield feed_list, len(index)

def run():
    if FLAGS.log_file_name:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open(FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 'data_reccon/all_data_pair.txt', FLAGS.w2v_file)
    # tf.debugging.set_log_device_placement(True)
    #with tf.device('/device:XLA_GPU:2'):
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')
    x_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len_bert]) 
    sen_len_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len]) 
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.float32) 
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    choice_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    if FLAGS.trans_type=='cross_road':
        y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * FLAGS.max_doc_len, FLAGS.n_class])
    else:
        y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * (FLAGS.window_size*2+1), FLAGS.n_class])
    placeholders = [x_bert, sen_len_bert, x, sen_len, doc_len, is_training, y_emotion, y_cause, y_pair, choice_len]  
    
    def get_bert_optimizer(loss_op):
        num_train_steps = int(1750 / FLAGS.batch_size * FLAGS.training_iter)
        # num_warmup_steps = int(num_train_steps * 0.1)
        num_warmup_steps = 291
        print('\n\nnum_warmup_steps {}\n\n'.format(num_warmup_steps))
        optimizer, run_lr = optimization.create_optimizer_dzx(loss_op, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)
        return optimizer, run_lr

    def init_from_bert_checkpoint():
        init_checkpoint = FLAGS.bert_base_dir + 'bert_model.ckpt'
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map) 
        
        tf.logging.info("**** Trainable Variables ****")
        for idx, var in enumerate(tvars):
            init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""
            tf.logging.info("var-index %s:  name = %s, shape = %s%s", idx, var.name, var.shape, init_string)

    def get_bert_tokenizer():
        do_lower_case = True
        tokenization.validate_case_matches_checkpoint(do_lower_case, FLAGS.bert_base_dir + 'bert_model.ckpt')
        return tokenization.FullTokenizer(vocab_file = FLAGS.bert_base_dir + 'vocab.txt', do_lower_case = do_lower_case)

    # Training Code Block
    print_info()
    #with tf.device('/device:XLA_GPU:0'):
    debug=1
    if debug== 1:
        pred_emo, pred_cause, pred_pair, reg = build_model(word_embedding, pos_embedding, x_bert, sen_len_bert, x, sen_len, doc_len, is_training, choice_len)
        print('build model done!\n')

        loss_emo = - tf.reduce_sum(y_emotion * tf.log(pred_emo)) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
        loss_cause = - tf.reduce_sum(y_cause * tf.log(pred_cause)) / tf.cast(tf.reduce_sum(y_cause), dtype=tf.float32)
        loss_pair = - tf.reduce_sum(y_pair * tf.log(pred_pair)) / tf.cast(tf.reduce_sum(y_pair), dtype=tf.float32)
        loss_op = loss_cause * FLAGS.cause + loss_emo * FLAGS.emo + loss_pair * FLAGS.pair + reg * FLAGS.l2_reg
        optimizer, run_lr = get_bert_optimizer(loss_op)
        true_y_emo_op = tf.argmax(y_emotion, 2)
        pred_y_emo_op = tf.argmax(pred_emo, 2)
        true_y_cause_op = tf.argmax(y_cause, 2)
        pred_y_cause_op = tf.argmax(pred_cause, 2)
        true_y_pair_op = y_pair
        pred_y_pair_op = pred_pair


    #with tf.device('/device:XLA_GPU:2'):
        tf_config = tf.ConfigProto()  
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            emo_list, cause_list, pair_list = [], [], []
            tokenizer = get_bert_tokenizer()
            for fold in range(1):
                # train for one fold
                print('############# fold {} begin ###############'.format(fold))
                init_from_bert_checkpoint()
                sess.run(tf.global_variables_initializer())

                # Data Code Block
                train_file_name = 'dailydialog_train.txt'
                valid_file_name = 'dailydialog_valid.txt'
                test_file_name = 'dailydialog_test.txt'
                # test_file_name = 'iemocap_test.txt'
                if FLAGS.trans_type=='cross_road':
                    tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x_bert, tr_sen_len_bert, tr_x, tr_sen_len, tr_doc_len, tr_choice_len = load_data_CR_Bert('data_reccon/'+train_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len)
                    de_doc_id, de_y_emotion, de_y_cause, de_y_pair, de_y_pairs, de_x_bert, de_sen_len_bert, de_x, de_sen_len, de_doc_len, de_choice_len = load_data_CR_Bert('data_reccon/'+valid_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len)
                    te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x_bert, te_sen_len_bert, te_x, te_sen_len, te_doc_len, te_choice_len = load_data_CR_Bert('data_reccon/'+test_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len)
                else:
                    tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x_bert, tr_sen_len_bert, tr_x, tr_sen_len, tr_doc_len, tr_pair_left_cnt, tr_choice_len = load_data_WC_Bert('data_reccon/'+train_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len, window_size = FLAGS.window_size)
                    de_doc_id, de_y_emotion, de_y_cause, de_y_pair, de_y_pairs, de_x_bert, de_sen_len_bert, de_x, de_sen_len, de_doc_len, de_pair_left_cnt, de_choice_len = load_data_WC_Bert('data_reccon/'+valid_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len, window_size = FLAGS.window_size)
                    te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x_bert, te_sen_len_bert, te_x, te_sen_len, te_doc_len, te_pair_left_cnt, te_choice_len = load_data_WC_Bert('data_reccon/'+test_file_name, tokenizer, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_sen_len, window_size = FLAGS.window_size)
                
                max_f1_emo, max_f1_cause, max_f1_pair = [-1.] * 3
                for i in range(FLAGS.training_iter):
                    start_time, step = time.time(), 1
                    test_results = []
                    def combine_result(x):
                        ret = []
                        for a in list(x):
                            ret.extend(list(a))
                        return np.array(ret) 
                    # train                        
                    print('epoch {}: cost time: {:.1f}s'.format(i, time.time()-start_time))
                    for train, _ in get_batch_data(tr_x_bert, tr_sen_len_bert, tr_x, tr_sen_len, tr_doc_len, 1, tr_y_emotion, tr_y_cause, tr_y_pair, FLAGS.batch_size, tr_choice_len):
                        _, run_lr_tmp, loss, pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch = sess.run(
                            [optimizer, run_lr, loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, true_y_emo_op, pred_y_pair_op, true_y_pair_op, doc_len], feed_dict=dict(zip(placeholders, train)))

                    ################## valid
                    valid_results = []
                    for valid, _ in get_batch_data(de_x_bert, de_sen_len_bert, de_x, de_sen_len, de_doc_len, 0., de_y_emotion, de_y_cause, de_y_pair, FLAGS.batch_size, choice_len=de_choice_len, test=True):
                        valid_results_tmp = sess.run(
                            [loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, true_y_emo_op, pred_pair, true_y_pair_op, doc_len], feed_dict=dict(zip(placeholders, valid)))
                        valid_results_tmp.append(de_y_cause)
                        valid_results.append(valid_results_tmp)
                    valid_results = list(zip(*valid_results))
                    loss = np.array(valid_results[0]).mean()
                    pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch, nonneutral = map(combine_result, valid_results[1:])
                    p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                    if f1 > max_f1_emo:
                        max_p_emo, max_r_emo, max_f1_emo = p, r, f1
                    print('VALID: emotion_prediction: p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    # print('VALID: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(max_p_emo, max_r_emo, max_f1_emo))
                    if FLAGS.trans_type=='cross_road':
                        p_f1, n_f1, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, de_y_cause, threshold = FLAGS.threshold)
                    else:
                        p_f1, n_f1, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, de_pair_left_cnt, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                    if f1 > max_f1_pair:
                        max_p_f1, max_n_f1, max_f1 = p_f1, n_f1, f1
                    print(f'VALID: pair_prediction:  pos_f1: {p_f1}, neg_f1: {n_f1}, avg_f1: {f1} ')
                    # print(f'VALID: max_p_f1: {max_p_f1}, max_n_f1:{max_n_f1}, max_f1:{max_f1}\n')

                    ################## test
                    test_results = []
                    for test, _ in get_batch_data(te_x_bert, te_sen_len_bert, te_x, te_sen_len, te_doc_len, 0., te_y_emotion, te_y_cause, te_y_pair, FLAGS.batch_size, choice_len=te_choice_len, test=True):
                        test_results_tmp = sess.run(
                            [loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, true_y_emo_op, pred_pair, true_y_pair_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                        test_results_tmp.append(te_y_cause)
                        test_results.append(test_results_tmp)
                    
                    test_results = list(zip(*test_results))
                    loss = np.array(test_results[0]).mean()
                    # pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch = map(combine_result, test_results[1:])
                    pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, pred_y_pair, true_y_pair, doc_len_batch, nonneutral = map(combine_result, test_results[1:])
                    
                    p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                    if f1 > max_f1_emo:
                        max_p_emo, max_r_emo, max_f1_emo = p, r, f1
                    print('TEST: emotion_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    # print('TEST: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(max_p_emo, max_r_emo, max_f1_emo))

                    # p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                    # if f1 > max_f1_cause:
                    #     max_p_cause, max_r_cause, max_f1_cause = p, r, f1
                    # print('cause_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    # print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_cause, max_r_cause, max_f1_cause))

                    if FLAGS.trans_type=='cross_road':
                        p_f1, n_f1, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, te_y_cause, threshold = FLAGS.threshold)
                        # p, r, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, threshold = FLAGS.threshold)
                    else:
                        p_f1, n_f1, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, te_pair_left_cnt, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                        # p, r, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, te_pair_left_cnt, threshold = FLAGS.threshold, window_size =FLAGS.window_size)
                    if f1 > max_f1_pair:
                        # max_p_pair, max_r_pair, max_f1_pair = p, r, f1
                        max_p_f1, max_n_f1, max_f1 = p_f1, n_f1, f1
                    # print('pair_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    # print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_pair, max_r_pair, max_f1_pair))
                    print(f'TEST: pair_prediction: pos_f1: {p_f1}, neg_f1: {n_f1}, avg_f1: {f1} ')
                    # print(f'TEST: max_p_f1: {max_p_f1}, max_n_f1:{max_n_f1}, max_f1:{max_f1}\n')

                print('Optimization Finished!\n')
                print('############# fold {} end ###############'.format(fold))
                

            print_time()
     
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO) 
    FLAGS.trans_type, FLAGS.trans_iter = 'cross_road', 0
    # FLAGS.log_file_name = 'log/ECPE-2D(Inter-EC(Bert))_1.log'.format(FLAGS.trans_iter)
    run()

    FLAGS.trans_type, FLAGS.trans_iter = 'window_constrained', 1
    # FLAGS.log_file_name = 'log/ECPE-2D(Inter-EC(Bert)+WC)_1.log'.format(FLAGS.trans_iter)
    run()

    FLAGS.trans_type, FLAGS.trans_iter = 'cross_road', 2
    # FLAGS.log_file_name = 'log/ECPE-2D(Inter-EC(Bert)+CR)_1.log'.format(FLAGS.trans_iter)
    run()


if __name__ == '__main__':
    tf.app.run()
