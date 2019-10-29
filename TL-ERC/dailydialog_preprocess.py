from multiprocessing import Pool
from pathlib import Path
from collections import OrderedDict
from urllib.request import urlretrieve
import os
import argparse
import tarfile
import pickle

from tqdm import tqdm
import pandas as pd, numpy as np

from utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
dailydialog_dir = datasets_dir.joinpath('dailydialog/')

# dailydialog_meta_dir = dailydialog_dir.joinpath('meta/')
# dialogs_dir = dailydialog_dir.joinpath('dialogs/')
GLOVE_DIR = ""

tokenizer = Tokenizer('spacy')

emo_classes = {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3, 
                'anger': 4, 'fear': 5, 'disgust':6}


def read_and_tokenize(dialog_path):
    """
    Read conversation
    Args:
        dialog_path (str): path of dialog (tsv format)
    Return:
        dialogs: (list of list of str) [dialog_length, sentence_length]
        users: (list of str); [2]
    """
    
    all_dialogs = []
    all_emotion_classes = []
    with open(dialog_path, 'r') as f:
        
        for line in tqdm(f):
            dialog = []
            emotions = []
            
            s = eval(line)
            for item in s['dialogue']:
                # print (item['text'])
                dialog.append(item['text'])
                emotions.append(emo_classes[item['emotion']])
                
            # print ('-'*30)
            dialog = [tokenizer(sentence) for sentence in dialog]
            
            #for k in range(1, len(dialog)):
            all_dialogs.append(dialog)
            all_emotion_classes.append(emotions)

    return all_dialogs, all_emotion_classes #, users


def pad_sentences(conversations, max_sentence_length=30, max_conversation_length=10):

    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    # [n_conversations, n_sentence (various), max_sentence_length]
    sentences = all_padded_sentences
    # [n_conversations, n_sentence (various)]
    sentence_length = all_sentence_length
    return sentences, sentence_length


def load_pretrained_glove(path):
    glv_vector = {}
    f = open(path, encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    return glv_vector


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=38) # Dont change this

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_vocab_frequency', type=int, default=5)

    # Multiprocess
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--glv_path_100', type=str, default='')
    parser.add_argument('--glv_path_300', type=str, default='')

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers
    
    glv_path_100 = args.glv_path_100
    glv_path_300 = args.glv_path_300
    


    def to_pickle(obj, path):
        with open(str(path), 'wb') as f:
            pickle.dump(obj, f)

    for split_type in ['train', 'test', 'valid']:
        
        print('Processing {a} dataset.'.format(a=split_type))
        split_data_dir = dailydialog_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)
        
        dialog_path = 'datasets/dailydialog/' + split_type + '.json'
        
        conversations, emotions = read_and_tokenize(dialog_path)
        
        shuffled_indices = np.arange(len(conversations))
        
        for j in range(10):
            np.random.shuffle(shuffled_indices)
            
        conversations = list(np.array(conversations)[shuffled_indices])
        emotions = list(np.array(emotions)[shuffled_indices])



        print ('Number of instances in {a} data: {b}.'.format(a=split_type, b=len(conversations)))

        conversation_length = [len(conversation) for conversation in conversations]


        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)


        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))
        to_pickle(emotions, split_data_dir.joinpath('labels.pkl'))

        if split_type == 'train':


            print('Save Vocabulary...')
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conversations)
            assert(GLOVE_DIR != "")
            vocab.update(GLOVE_DIR, max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(dailydialog_dir.joinpath('word2id.pkl'),
                         dailydialog_dir.joinpath('id2word.pkl'),
                         dailydialog_dir.joinpath('word_emb.pkl'))
            

        print('Done!')

