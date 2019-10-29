from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from util import to_var, time_desc_decorator
import os
from tqdm import tqdm
from math import isnan
import re
import math
import pickle
import gensim
from sklearn.metrics import classification_report


class Solver(object):
    def __init__(self, config, train_data_loader, valid_data_loader, test_data_loader, is_train=True, model=None):

        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:

            self.model = getattr(models, self.config.model)(self.config)

            # orthogonal initialiation for hidden weights
            # input gate bias for GRUs
            if self.config.mode == 'train' and self.config.checkpoint is None:

                # Make later layers require_grad = False
                for name, param in self.model.named_parameters():
                    if "encoder.encoder.layer" in name:
                        layer_num = int(name.split("encoder.encoder.layer.")[-1].split(".")[0])
                        if layer_num >= (self.config.num_bert_layers):
                            param.requires_grad = False


                print('Parameter initiailization')
                for name, param in self.model.named_parameters(): 
                    if ('weight_hh' in name) and ("encoder.encoder" not in name):
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                # Final list
                for name, param in self.model.named_parameters():
                    print('\t' + name, param.requires_grad)


        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.load_checkpoint:
            self.load_model(self.config.load_checkpoint)

        if self.is_train:
            self.optimizer = self.config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate)


    def load_model(self, checkpoint):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')

        pretrained_dict = torch.load(checkpoint)
        model_dict =self.model.state_dict()

        # 1. filter out unnecessary keys
        filtered_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if (k in model_dict) and ("embedding" not in k) and ("context" in k) and ("ih" not in k):
                filtered_pretrained_dict[k]=v

        print(f"Filtered pretrained dict: {filtered_pretrained_dict.keys()}")

        # 2. overwrite entries in the existing state dict
        model_dict.update(filtered_pretrained_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)


    @time_desc_decorator('Training Start!')
    def train(self):
        min_val_loss = np.inf
        patience_counter=0
        best_epoch = -1
        
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i

            batch_loss_history = []
            predictions, ground_truth = [], []
            self.model.train()
            n_total_words = 0
            before_gradient = None

            for batch_i, (conversations, labels, conversation_length, sentence_length, type_ids, masks) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                input_conversations = conversations

                # flatten input and target conversations
                input_sentences = [sent for conv in input_conversations for sent in conv]
                input_labels = [label for utt in labels for label in utt]
                input_sentence_length = [l for len_list in sentence_length for l in len_list]
                input_conversation_length = [l for l in conversation_length]
                input_masks = [mask for conv in masks for mask in conv]
                orig_input_labels = input_labels

                # transfering the input to cuda
                input_sentences = to_var(torch.LongTensor(input_sentences))
                input_labels = to_var(torch.LongTensor(input_labels))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                input_masks = to_var(torch.LongTensor(input_masks))

                # reset gradient
                self.optimizer.zero_grad()

                sentence_logits = self.model(
                    input_sentences,
                    input_sentence_length,
                    input_conversation_length,
                    input_masks)

                present_predictions = list(np.argmax(sentence_logits.detach().cpu().numpy(), axis=1))

                loss_function = nn.CrossEntropyLoss()
                batch_loss = loss_function(sentence_logits, input_labels)

                predictions += present_predictions
                ground_truth += orig_input_labels

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())

                if batch_i % self.config.print_every == 0:
                    tqdm.write(
                        f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item()}')

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()


            epoch_loss = np.mean(batch_loss_history)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}'
            print(print_str)
            
            self.w_train_f1 = self.print_metric(ground_truth, predictions, "train")


            self.validation_loss, self.w_valid_f1, valid_predictions = self.evaluate(self.valid_data_loader, mode="valid")
            self.test_loss, self.w_test_f1, test_predictions = self.evaluate(self.test_data_loader, mode="test")

            print(self.epoch_loss, self.w_train_f1, self.w_valid_f1, self.w_test_f1)

            IMPROVED = False
            if self.validation_loss < min_val_loss:
                IMPROVED = True
                min_val_loss = self.validation_loss
                best_test_loss = self.test_loss
                best_test_f1_w = self.w_test_f1
                best_epoch = (self.epoch_i+1)

            if (not IMPROVED):
                patience_counter+=1
            else:
                patience_counter = 0
            print(f'Patience counter: {patience_counter}')
            if (patience_counter > self.config.patience):
                break


        return best_test_loss, best_test_f1_w, best_epoch



    def evaluate(self, data_loader, mode=None):
        assert(mode is not None)

        self.model.eval()
        batch_loss_history, predictions, ground_truth = [], [], []
        for batch_i, (conversations, labels, conversation_length, sentence_length, type_ids, masks) in enumerate(data_loader):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            input_conversations = conversations

            # flatten input and target conversations
            input_sentences = [sent for conv in input_conversations for sent in conv]
            input_labels = [label for conv in labels for label in conv]
            input_sentence_length = [l for len_list in sentence_length for l in len_list]
            input_conversation_length = [l for l in conversation_length]
            input_masks = [mask for conv in masks for mask in conv]
            orig_input_labels = input_labels

            with torch.no_grad():
                # transfering the input to cuda
                input_sentences = to_var(torch.LongTensor(input_sentences))
                input_labels = to_var(torch.LongTensor(input_labels))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                input_masks = to_var(torch.LongTensor(input_masks))

            sentence_logits = self.model(
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                input_masks)

            present_predictions = list(np.argmax(sentence_logits.detach().cpu().numpy(), axis=1))
            
            loss_function = nn.CrossEntropyLoss()
            batch_loss = loss_function(sentence_logits, input_labels)

            predictions += present_predictions
            ground_truth += orig_input_labels

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())

        epoch_loss = np.mean(batch_loss_history)

        print_str = f'{mode} loss: {epoch_loss:.3f}\n'

        w_f1_score = self.print_metric(ground_truth, predictions, mode)
        return epoch_loss, w_f1_score, predictions
    

    
    def print_metric(self, y_true, y_pred, mode):

        if mode in ["train", "test"]:
            print(mode)
            if (self.config.data == "dailydialog"):
                print(classification_report(y_true, y_pred, labels=[1,2,3,4,5,6], digits=4))
            else:
                print(classification_report(y_true, y_pred, digits=4))
        

        if (self.config.data == "dailydialog"):
            weighted_fscore = classification_report(y_true, y_pred, labels=[1,2,3,4,5,6], output_dict=True, digits=4)["weighted avg"]["f1-score"]
        else:
            weighted_fscore = classification_report(y_true, y_pred, output_dict=True, digits=4)["weighted avg"]["f1-score"]

        return weighted_fscore
