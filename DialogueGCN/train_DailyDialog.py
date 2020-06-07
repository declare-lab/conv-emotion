import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import DailyDialogueDataset2
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel, DialogueGCN_DailyModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_DailyDialogue_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = DailyDialogueDataset2('train', path)
    testset = DailyDialogueDataset2('test', path)
    validset = DailyDialogueDataset2('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # import ipdb;ipdb.set_trace()
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
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

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]

# Only modified graph model
def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    # Here, You should mask the label 'no_emotion'
    # So you should remember the pre-processing label mapping
    # My 'no_emotion' label matched to '3'
    # Follow the same metric settings in DialogueRNN
    avg_fscore = round(f1_score(labels, preds, average='micro', labels=[0, 1, 2, 4, 5, 6]) * 100, 2)
    # Add precision and recall
    precision = round(precision_score(labels, preds, average='micro', labels=[0, 1, 2, 4, 5, 6]) * 100, 2)
    recall = round(recall_score(labels, preds, average='micro', labels=[0, 1, 2, 4, 5, 6]) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, precision, recall


if __name__ == '__main__':

    path = './saved/DailyDialog/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=False,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

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

    n_classes = 7
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    # change D_m into
    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    kernel_sizes = [3, 4, 5]
    glv_pretrained = np.load(open('dailydialog/glv_embedding_matrix2', 'rb'))
    vocab_size, embedding_dim = glv_pretrained.shape
    if args.graph_model:
        seed_everything()

        model = DialogueGCN_DailyModel(args.base_model,
                                       D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                       n_speakers=2,
                                       max_seq_len=110,
                                       window_past=args.windowp,
                                       window_future=args.windowf,
                                       vocab_size=vocab_size,
                                       n_classes=n_classes,
                                       listener_state=args.active_listener,
                                       context_attention=args.attention,
                                       dropout=args.dropout,
                                       nodal_attention=args.nodal_attention,
                                       no_cuda=args.no_cuda
                                       )
        model.init_pretrained_embeddings(glv_pretrained)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print('Basic Dialog RNN Model.')


        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    # for daily_dialog_bert2.pkl
    # In the pre-processing, you should remember the label mapping.
    # Here is my label mapping and weights for the dailydialogue
    loss_weights = torch.FloatTensor([1 / 0.0017,
                                      1 / 0.0034,
                                      1 / 0.1251,
                                      1 / 0.831,
                                      1 / 0.0099,
                                      1 / 0.0177,
                                      1 / 0.0112])
    if args.class_weight:
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.class_weight:
        train_loader, valid_loader, test_loader = get_DailyDialogue_loaders('dailydialog/daily_dialogue2.pkl',
                                                                            batch_size=batch_size, num_workers=0)
    else:
        train_loader, valid_loader, test_loader = get_DailyDialogue_loaders('dailydialog/daily_dialogue2.pkl',
                                                                            batch_size=batch_size, num_workers=0)
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    all_precision, all_recall = [], []

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, train_precision, train_recall = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, valid_precision, valid_recall = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_precision, test_recall = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda)
            all_fscore.append(test_fscore)
            all_precision.append(test_precision)
            all_recall.append(test_recall)
            # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e,
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)
            all_fscore.append(test_fscore)
            # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, train_precision: {}, train_recall: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, valid_precision: {}, valid_recall: {}, test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, train_precision, train_recall, valid_loss, valid_acc,
                   valid_fscore, valid_precision, valid_recall, test_loss, test_acc, test_fscore, test_precision,
                   test_recall, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    index_max = all_fscore.index(max(all_fscore))
    print('F-Score:', all_fscore[index_max])
    print('Precision:', all_precision[index_max])
    print('Recall:', all_recall[index_max])
