import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from model import BiE2EModel,UnMaskedWeightedNLLLoss

from torchtext import data, vocab
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator, Pipeline

from keras.utils import to_categorical

import spacy
spacy_en = spacy.load('en')

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def tokenizer(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def convert_token(token, *args):
    return token-1

def get_E2E_loaders(path, valid=0.1, batch_size=32):
    utterance = data.Field(tokenize=tokenizer, lower=True)
    label     = data.Field(sequential=False, postprocessing=Pipeline(convert_token=convert_token))
    id        = data.Field(use_vocab=False,sequential=False)
    fields = [('id', id),
              ('turn1', utterance),
              ('turn2', utterance),
              ('turn3', utterance),
              ('label', label)]

    train = data.TabularDataset('{}/train.txt'.format(path),
                                format='tsv',
                                fields=fields,
                                skip_header=True)
    valid = data.TabularDataset('{}/valid.txt'.format(path),
                                format='tsv',
                                fields=fields,
                                skip_header=True)

    test = data.TabularDataset('{}/test.txt'.format(path),
                                format='tsv',
                                fields=fields,
                                skip_header=True)
    vectors = vocab.Vectors(name='emojiplusglove.txt', cache='/media/backup/nlp-cic/DialogueRNN/')
    utterance.build_vocab(train, valid, test, vectors=vectors)
    #utterance.build_vocab(train, valid, test, vectors='glove.840B.300d')
    label.build_vocab(train)
    train_iter = BucketIterator(train,
                                  train=True,
                                  batch_size=batch_size,
                                  sort_key=lambda x: len(x.turn3),
                                  device=torch.device(0))
    valid_iter = BucketIterator(valid,
                                  batch_size=batch_size,
                                  sort_key=lambda x: len(x.turn3),
                                  device=torch.device(0))
    test_iter = BucketIterator(test,
                                  batch_size=batch_size,
                                  sort_key=lambda x: len(x.turn3),
                                  device=torch.device(0))
    return train_iter, valid_iter, test_iter,\
            utterance.vocab.vectors if not args.cuda else utterance.vocab.vectors.cuda(),\
            label.vocab.itos

def train_or_eval_model(model, embeddings, dataloader, epoch, loss_function=None, optimizer=None, train=False, valid=False, test=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    # assert not train or optimizer!=None
    #umask = torch.FloatTensor([[1,1,1]]).type(T1.type())
    #umask = umask.expand( T1.size(1),-1)
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        log_prob = model(data,True) # batch, n_classes
        lp_ = log_prob # batch, n_classes
        # import ipdb;ipdb.set_trace()
        if train or valid or test:
            labels_ = data.label # batch
            loss = loss_function(lp_, labels_)
            losses.append(loss.item())

        pred_ = torch.argmax(lp_,1) # batch
        preds.append(pred_.data.cpu().numpy())
        if train or valid or test:
            labels.append(labels_.data.cpu().numpy())
            masks.append(data.turn1.size(1))
        else:
            masks.append(data.id.data.cpu().numpy())

        if train:
            loss.backward()
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if train or valid or test:
        if preds!=[]:
            # import ipdb;ipdb.set_trace()
            preds  = np.concatenate(preds)
            labels = np.concatenate(labels)
        else:
            return float('nan'), float('nan'), [], [], float('nan')

        avg_loss = round(np.sum(losses)/np.sum(masks),4)
        avg_accuracy = round(accuracy_score(labels,preds)*100,2)
        _,_,_,avg_fscore = get_metrics(labels,preds)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore
    else:
        preds  = np.concatenate(preds)
        masks  = np.concatenate(masks)
        return masks, preds
def get_metrics(discretePredictions, ground,n_classes=4):
    
    discretePredictions = to_categorical(discretePredictions,4)
    ground = to_categorical(ground,4)
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    accuracy = np.mean(discretePredictions==ground)
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, n_classes):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=15, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes  = 4
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_emb = 300
    D_m   = 200
    D_g   = 150
    D_p   = 150
    D_e   = 100
    D_h   = 100

    D_a = 100 # concat attention

    #model = BiE2EModel(D_emb, D_m, D_g, D_p, D_e, D_h,
    #                 n_classes=n_classes,
    #                 listener_state=args.active_listener,
    #                 context_attention=args.attention,
    #                 dropout_rec=args.rec_dropout,
    #                 dropout=args.dropout)
    #if cuda:
    #    model.cuda()
    loss_weights = torch.FloatTensor([
                                        2, 1, 1 , 1
                                        ])
    if args.class_weight:
        loss_function  = UnMaskedWeightedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = UnMaskedWeightedNLLLoss()
    #optimizer = optim.Adam(model.parameters(),
    #                       lr=args.lr,
    #                       weight_decay=args.l2)

    train_loader, valid_loader, test_loader, embeddings, id2label =\
            get_E2E_loaders('./semeval19_emocon',
                            valid=0.1,
                            batch_size=batch_size)
    #optimizer = optim.Adam(model.parameters(),
    #                       lr=args.lr,
    #                       weight_decay=args.l2)

    model = BiE2EModel(D_emb, D_m, D_g, D_p, D_e, D_h, embeddings,
                     n_classes=n_classes,
                     listener_state=args.active_listener,
                     context_attention=args.attention,
                     dropout_rec=args.rec_dropout,
                     dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    if cuda:
        model.cuda()

    best_loss, best_f1, best_pred, best_val_pred, best_val_label, best_ids =\
            None, None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,train_fscore = train_or_eval_model(model, embeddings,
                                               train_loader, e, loss_function, optimizer, train=True)
        valid_loss, valid_acc, valid_label, valid_pred, val_fscore = train_or_eval_model(model, embeddings, valid_loader, e, loss_function, valid=True)
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model(model, embeddings, test_loader, e, loss_function, test=True)

        if best_loss == None or best_loss > valid_loss:
            best_loss, best_f1, best_pred, best_test_pred, best_test_label =\
                    valid_loss, test_fscore, test_pred+1, test_pred, test_label
            best_valid_f1, best_pred, best_valid_pred, best_valid_label =\
                    val_fscore, valid_pred+1, valid_pred, valid_label

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
            writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)
        print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore, test_loss, test_acc, test_fscore, \
                        round(time.time()-start_time,2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} fscore {}'.format(best_loss, round(best_f1,2)))
    print(classification_report(best_test_label,best_test_pred,digits=4))
    print(confusion_matrix(best_test_label,best_test_pred))

    print('Valid performance..')
    print('Loss {} fscore {}'.format(best_loss, round(best_valid_f1,2)))
    print(classification_report(best_valid_label,best_valid_pred,digits=4))
    print(confusion_matrix(best_valid_label,best_valid_pred))
    #with open('./semeval19_emocon/test.txt','w') as f:
    #    f.write('id\tturn1\tturn2\tturn3\tlabel\n')
    #    for id, label in zip(best_ids, best_pred):
    #        f.write('{}\tdummy1\tdummy2\tdummy3\t{}\n'.format(id, id2label[label]))
    # with open('best_attention.p','wb') as f:
    #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)
