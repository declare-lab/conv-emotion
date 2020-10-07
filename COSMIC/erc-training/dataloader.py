import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        
        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]
    

class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping = 
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('meld/meld_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor(self.speakers[vid]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]


class DailyDialogueRobertaCometDataset(Dataset):

    def __init__(self, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]

class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''
        
        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
        = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
        
        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('emorynlp/emorynlp_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]
            
        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]
