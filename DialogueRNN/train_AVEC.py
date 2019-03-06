import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

import pandas as pd

from model import AVECModel, MaskedMSELoss
from dataloader import AVECDataset


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = range(size)
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_AVEC_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = AVECDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = AVECDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label =\
                                [d.cuda() for d in data] if cuda else data
        pred = model(textf, qmask) # batch*seq_len
        labels_ = label.view(-1) # batch*seq_len
        umask_ = umask.view(-1) # batch*seq_len
        loss = loss_function(pred, labels_, umask_)

        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask_.cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    mae = round(mean_absolute_error(labels,preds,sample_weight=masks),4)
    pred_lab = pd.DataFrame(list(filter(lambda x: x[2]==1, zip(labels, preds, masks))))
    pear = round(pearsonr(pred_lab[0], pred_lab[1])[0], 4)
    return avg_loss, mae, pear, labels, preds, masks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.0,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute')
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

    D_m = 100
    D_g = 100
    D_p = 100
    D_e = 100
    D_h = 100

    D_a = 100 # concat attention

    model = AVECModel(D_m, D_g, D_p, D_e, D_h,
                    attr=args.attribute,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)
    if cuda:
        model.cuda()
    loss_function = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader =\
            get_AVEC_loaders('./AVEC_features/AVEC_features_{}.pkl'.format(args.attribute),
                                valid=0.0,
                                batch_size=batch_size,
                                num_workers=2)

    best_loss, best_label, best_pred, best_mask, best_pear = None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_mae, train_pear,_,_,_ = train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_mae, valid_pear,_,_,_ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_mae, test_pear, test_label, test_pred, test_mask = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_pear =\
                    test_loss, test_label, test_pred, test_mask, test_pear

        if args.tensorboard:
            writer.add_scalar('test: loss',test_loss,e)
            writer.add_scalar('train: loss',train_loss,e)
            writer.add_scalar('test: mae',test_mae,e)
            writer.add_scalar('train: mae',train_mae,e)
            writer.add_scalar('test: pear',test_pear,e)
            writer.add_scalar('train: pear',train_pear,e)
        print('epoch {} train_loss {} train_mae {} train_pear {} valid_loss {} valid_mae {} valid_pear {} test_loss {} test_mae {} test_pear {} time {}'.\
                format(e+1, train_loss, train_mae, train_pear, valid_loss, valid_mae,\
                        valid_pear, test_loss, test_mae, test_pear, round(time.time()-start_time,2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} MAE {} r {}'.format(best_loss,
                                 round(mean_absolute_error(best_label,best_pred,sample_weight=best_mask),4),
                                 best_pear))
