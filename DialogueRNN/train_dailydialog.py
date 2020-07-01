import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np, pickle, time, argparse
from model import DailyDialogueModel, MaskedNLLLoss
from dataloader import DailyDialoguePadCollate, DailyDialogueDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support


def get_DailyDialogue_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    
    trainset = DailyDialogueDataset('train', path)
    testset = DailyDialogueDataset('test', path)
    validset = DailyDialogueDataset('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn = DailyDialoguePadCollate(dim=0),
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn = DailyDialoguePadCollate(dim=0),
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn = DailyDialoguePadCollate(dim=0),
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def process_data_loader(data):
    
    input_sequence, qmask, umask, act_labels, emotion_labels, max_sequence_lengths, _ = data
    input_sequence = input_sequence[:, :, :max(max_sequence_lengths)]
    
    input_sequence, qmask, umask = input_sequence.cuda(), qmask.cuda(), umask.cuda()
    # act_labels = act_labels.cuda()
    emotion_labels = emotion_labels.cuda()
    
    return [input_sequence, qmask, umask, emotion_labels]



def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
        
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        input_sequence, qmask, umask, label = process_data_loader(data)
        log_prob, alpha, alpha_f, alpha_b = model(input_sequence, qmask, umask)
        
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='micro', labels=[0,2,3,4,5,6])*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--cnn_filters', type=int, default=50,
                        help='Number of cnn filters for cnn feature extraction')
    parser.add_argument('--cnn_output_size', type=int, default=100,
                        help='feature size from cnn layer')
    parser.add_argument('--cnn_dropout', type=float, default=0.5, metavar='cnn_dropout',
                        help='cnn dropout rate')
    args = parser.parse_args()

    print(args)
    
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes  = 7
    cuda       = args.cuda
    n_epochs   = args.epochs
    
    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    
    kernel_sizes = [1,2,3]
    
    glv_pretrained = np.load(open('dailydialog/glv_embedding_matrix', 'rb'))
    vocab_size, embedding_dim = glv_pretrained.shape
    # glv_pretrained[0, :] = np.random.rand(embedding_dim)
    model = DailyDialogueModel(D_m, D_g, D_p, D_e, D_h, vocab_size=vocab_size, n_classes=7, 
                               embedding_dim=embedding_dim,
                               cnn_output_size=args.cnn_output_size,
                               cnn_filters=args.cnn_filters, 
                               cnn_kernel_sizes=kernel_sizes,
                               cnn_dropout=args.cnn_dropout,
                               listener_state=args.active_listener,
                               context_attention=args.attention,
                               dropout_rec=args.rec_dropout,
                               dropout=args.dropout)
    model.init_pretrained_embeddings(glv_pretrained)    
    if cuda:
        model.cuda()
        
        
    loss_weights = torch.FloatTensor([1.2959,0.7958,0.8276,1.4088,0.9560,1.0575,0.6585])
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr,
                           weight_decay=args.l2)
    

    train_loader, valid_loader, test_loader = get_DailyDialogue_loaders('dailydialog/daily_dialogue.pkl', 
                                                                        batch_size=batch_size, num_workers=0)
    
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore,_= train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc, _,_,_,val_fscore,_= train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn =                    test_loss, test_label, test_pred, test_mask, attentions

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
            writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)
        print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} test_loss {} test_acc {} test_fscore {} time {}s'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} F1-score {}'.format(best_loss,
                                     round(f1_score(best_label,best_pred,sample_weight=best_mask, average='micro', labels=[0,2,3,4,5,6])*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,labels=[0,2,3,4,5,6],digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
