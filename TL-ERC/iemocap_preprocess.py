# Preprocess iemocap conversation emotion dataset

import argparse
import pickle
import random
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils import Vocab, Tokenizer, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
iemocap_dir = datasets_dir.joinpath('iemocap/')
iemocap_pickle = iemocap_dir.joinpath("IEMOCAP_features_raw.pkl")
GLOVE_DIR = ""

# Tokenizer
tokenizer = Tokenizer('spacy')

class IEMOCAP:
    '''
    label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    '''
    def load_iemocap_data(self):
        _, self.videoSpeakers, self.videoLabels, _, _, _, self.videoSentence, trainVid, self.testVid = pickle.load(
            open(iemocap_pickle, "rb"), encoding="latin1")

        self.trainVid, self.valVid = train_test_split(
            list(trainVid), test_size=.2, random_state=1227)
        
        self.vids = {"train":self.trainVid, "valid":self.valVid, "test":self.testVid}
        
        # Calculating maximum sentence length
        self.max_conv_length = max([len(self.videoSentence[vid]) for vid in self.trainVid])
        

def tokenize_conversation(lines):
    sentence_list = [tokenizer(line) for line in lines]
    return sentence_list


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
        sentence_length = [min(len(sentence) + 1, max_sentence_length)  # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Load the dataset
    iemocap = IEMOCAP()
    iemocap.load_iemocap_data()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_vocab_frequency', type=int, default=5)

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = iemocap.max_conv_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency

    

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    for split_type in ['train', 'valid', 'test']:
        conv_sentences = [iemocap.videoSentence[vid] for vid in iemocap.vids[split_type]]
        conv_labels = [iemocap.videoLabels[vid]
                       for vid in iemocap.vids[split_type]]


        print(f'Processing {split_type} dataset...')
        split_data_dir = iemocap_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)
        
        conv_sentences = list([tokenize_conversation(conv) for conv in conv_sentences])
        conversation_length = [min(len(conv), max_conv_len)
                               for conv in conv_sentences]

        # fix labels as per conversation_length
        for idx, conv_len in enumerate(conversation_length):
            conv_labels[idx]=conv_labels[idx][:conv_len]

        
        sentences, sentence_length = pad_sentences(
            conv_sentences,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        for sentence_len, label in zip(conversation_length, conv_labels):
            assert(sentence_len ==len(label))

        
        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath(
            'conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(conv_labels, split_data_dir.joinpath('labels.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath(
            'sentence_length.pkl'))
        to_pickle(iemocap.vids[split_type], split_data_dir.joinpath('video_id.pkl'))

        if split_type == 'train':

            print('Save Vocabulary...')
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conv_sentences)

            assert(GLOVE_DIR != "")
            vocab.update(GLOVE_DIR, max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(iemocap_dir.joinpath('word2id.pkl'),
                         iemocap_dir.joinpath('id2word.pkl'),
                         iemocap_dir.joinpath('word_emb.pkl'))