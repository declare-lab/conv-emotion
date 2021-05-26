import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = '../data_reccon'
# TRAIN_FILE = 'fold%s_train.json'
# VALID_FILE = 'fold%s_valid.json'
# TEST_FILE  = 'fold%s_test.json'
TRAIN_FILE = 'dailydialog_train.json'
VALID_FILE = 'dailydialog_valid.json'
TEST_FILE  = 'dailydialog_test.json'
# TEST_FILE  = 'iemocap_test.json'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = 'bert-base-uncased'
        self.feat_dim = 768

        self.gnn_dims = '192'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 15
        self.lr = 1e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

