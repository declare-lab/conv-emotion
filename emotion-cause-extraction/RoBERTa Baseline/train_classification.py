import numpy as np, pandas as pd
import json, os, logging, pickle, argparse
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel

if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate') 
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob| robl')
    parser.add_argument('--fold', type=int, default=1, metavar='F', help='which fold')
    parser.add_argument('--context', action='store_true', default=False, help='use context')
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    args = parser.parse_args()

    print(args)
    
    model_family = {'rob': 'roberta', 'robl': 'roberta'}
    model_id = {'rob': 'roberta-base', 'robl': 'roberta-large'}
    model_exact_id = {'rob': 'roberta-base', 'robl': 'roberta-large'}
    
    
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = args.model
    fold = str(args.fold)
    context = args.context
    cuda = args.cuda
    dataset = 'dailydialog'
    
    if context == False:
        max_seq_length = 300
    else:
        max_seq_length = 512
        
    if context == False:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-cls-without-context-fold' + fold + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-cls-without-context-fold' + fold + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-cls-without-context-fold' + fold + '/test_predictions.pkl'
        x_train = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_train_without_context.csv')
        x_valid = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_valid_without_context.csv')
        x_test  = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_test_without_context.csv')
    else:
        save_dir    = 'outputs/' + model_id[model] + '-dailydialog-cls-with-context-fold' + fold + '/'
        result_file = 'outputs/' + model_id[model] + '-dailydialog-cls-with-context-fold' + fold + '/results.txt'
        dump_file   = 'outputs/' + model_id[model] + '-dailydialog-cls-with-context-fold' + fold + '/test_predictions.pkl'
        x_train = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_train_with_context.csv')
        x_valid = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_valid_with_context.csv')
        x_test  = pd.read_csv('data/subtask2/fold' + fold + '/dailydialog_classification_test_with_context.csv')
    
    if fold == '1':
        num_steps = int(27915/batch_size)
    else:
        num_steps = int(25697/batch_size)
    
    train_args = {
        'fp16': False,
        'overwrite_output_dir': True, 
        'max_seq_length': max_seq_length,
        'learning_rate': lr,
        'sliding_window': False,
        'output_dir': save_dir,
        'best_model_dir': save_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': num_steps,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': 500000,
        'train_batch_size': batch_size,
        'num_train_epochs': epochs
    }
    
    cls_model = ClassificationModel(model_family[model], model_exact_id[model], args=train_args, cuda_device=cuda)
    cls_model.train_model(x_train, eval_df=x_valid)
    
    cls_model = ClassificationModel(model_family[model], save_dir + 'best_model/', args=train_args, cuda_device=cuda)
    result, model_outputs, wrong_predictions = cls_model.eval_model(x_test)
    
    preds = np.argmax(model_outputs, 1)
    labels = x_test['labels']

    r = str(classification_report(labels, preds, digits=4))
    print (r)
    
    rf = open('results/dailydialog_classification.txt', 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*54 + '\n')    
    rf.close()
    
    rf = open(result_file, 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*54 + '\n')    
    rf.close()
    
    pickle.dump([result, model_outputs, wrong_predictions], open(dump_file, 'wb'))
    
