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

from model import E2EModel

from torchtext import data
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator, Pipeline

import spacy
spacy_en = spacy.load('en')

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

    utterance.build_vocab(train, valid, test, vectors='glove.840B.300d')
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
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        log_prob = model(data, embeddings) # batch, n_classes
        lp_ = log_prob # batch, n_classes
        # import ipdb;ipdb.set_trace()
        if train or valid:
            labels_ = data.label # batch
            loss = loss_function(lp_, labels_)
            losses.append(loss.item())

        pred_ = torch.argmax(lp_,1) # batch
        preds.append(pred_.data.cpu().numpy())
        if train or valid:
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
        avg_fscore = round(f1_score(labels,preds,average='micro')*100,2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore
    else:
        preds  = np.concatenate(preds)
        masks  = np.concatenate(masks)
        return masks, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
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
    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_emb = 300
    D_m   = 200
    D_g   = 150
    D_p   = 150
    D_e   = 100
    D_h   = 100

    D_a = 100 # concat attention

    model = E2EModel(D_emb, D_m, D_g, D_p, D_e, D_h,
                     n_classes=n_classes,
                     listener_state=args.active_listener,
                     context_attention=args.attention,
                     dropout_rec=args.rec_dropout,
                     dropout=args.dropout)
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])
    loss_function = nn.NLLLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader, embeddings, id2label =\
            get_E2E_loaders('./semeval19_emocon',
                            valid=0.1,
                            batch_size=batch_size)

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
    #with open('./semeval19_emocon/test.txt','w') as f:
    #    f.write('id\tturn1\tturn2\tturn3\tlabel\n')
    #    for id, label in zip(best_ids, best_pred):
    #        f.write('{}\tdummy1\tdummy2\tdummy3\t{}\n'.format(id, id2label[label]))
    # with open('best_attention.p','wb') as f:
    #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)
