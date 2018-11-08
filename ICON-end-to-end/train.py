#Please use python 3.5 or above
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import io
import sys
import os

from data_helper import DataHelper
from model import ICON

# Selecting the GPU to work on
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Desired graphics card config
session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))


def pad_batch(data, remainder_length, dtype):
    new_shape = list(data.shape)
    new_shape[0]=remainder_length
    new_shape = tuple(new_shape)
    return np.concatenate((data, np.zeros(new_shape, dtype=dtype)), axis=0)


def train_model(config, dataQueries, dataOwnHistories, dataOtherHistories, labels, embeddingMatrix, batches):

    print("Training model on entire data")

    with tf.Graph().as_default():
        tf.set_random_seed(1234) # Graph level random seed
        sess = tf.Session(config=session_conf) # Defining the session of the Graph
        with sess.as_default():

            model = ICON(config, embeddingMatrix, session=sess)

            for t in range(1, config["num_epochs"]+1):

                # Annealing of the learning rate
                if t - 1 <= config["anneal_stop_epoch"]:
                    anneal = 2.0 ** ((t-1) // config["anneal_rate"])
                else:
                    anneal = 2.0 ** (config["anneal_stop_epoch"] // config["anneal_rate"])
                lr = config["learning_rate"] / anneal

                # Shuffling the batches in each epoch
                np.random.shuffle(batches)

                total_cost = 0.0
                for start, end in batches:
                    query = dataQueries[start:end]
                    ownHistory = dataOwnHistories[start:end]
                    otherHistory = dataOtherHistories[start:end]
                    answers = labels[start:end]
                    
                    if query.shape[0] < config["batch_size"]:
                        remainder_length = config["batch_size"]-query.shape[0]
                        query = pad_batch(query, remainder_length, np.float32)
                        ownHistory = pad_batch(ownHistory, remainder_length, np.float32)
                        otherHistory = pad_batch(otherHistory, remainder_length, np.float32)
                        answers = pad_batch(answers, remainder_length, np.float32)

                    cost_t = model.batch_fit(query, ownHistory, otherHistory, answers)
                    total_cost += cost_t
                print(total_cost)
            return model

def predict_model(config, model, dataQueries, dataOwnHistories, dataOtherHistories, batches):

    preds=[]
    for start, end in batches:
        query = dataQueries[start:end]
        ownHistory = dataOwnHistories[start:end]
        otherHistory = dataOtherHistories[start:end]

        
        if query.shape[0] < config["batch_size"]:
            remainder_length = config["batch_size"]-query.shape[0]
            query = pad_batch(query, remainder_length, np.float32)
            ownHistory = pad_batch(ownHistory, remainder_length, np.float32)
            otherHistory = pad_batch(otherHistory, remainder_length, np.float32)

        preds += list(model.predict(query, ownHistory, otherHistory))

    return preds[:len(dataQueries)]


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
    

    ####################  Pre-processing  #############################
    # Data Helper object
    datahelper = DataHelper(config)


    # Loading the data        
    print("Processing training data...")
    trainIndices, trainQueries, trainOwnHistories, trainOtherHistories, trainTexts, labels = datahelper.preprocessData(config["train_data_path"], mode="train")

    print("Processing test data...")
    testIndices, testQueries, testOwnHistories, testOtherHistories, testTexts = datahelper.preprocessData(config["test_data_path"], mode="test")

    # Size of data
    n_train = len(trainIndices)
    n_test = len(testIndices)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=config["max_nb_words"])
    tokenizer.fit_on_texts(trainTexts)
    trainQueriesSequences = tokenizer.texts_to_sequences(trainQueries)
    testQueriesSequences = tokenizer.texts_to_sequences(testQueries)
    trainOwnHistoriesSequences = tokenizer.texts_to_sequences(trainOwnHistories)
    testOwnHistoriesSequences = tokenizer.texts_to_sequences(testOwnHistories)
    trainOtherHistoriesSequences = tokenizer.texts_to_sequences(trainOtherHistories)
    testOtherHistoriesSequences = tokenizer.texts_to_sequences(testOtherHistories)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = datahelper.getEmbeddingMatrix(wordIndex)

    ####################  Training  #############################


    # Prepare training data
    dataQueries = pad_sequences(trainQueriesSequences, maxlen=config["max_sequence_length"])

    dataOwnHistories = datahelper.prepare_history(trainOwnHistoriesSequences, mode="own", maxlen=config["max_sequence_length"])
    dataOtherHistories = datahelper.prepare_history(trainOtherHistoriesSequences, mode="other", maxlen=config["max_sequence_length"])
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", dataQueries.shape, dataOwnHistories.shape, dataOtherHistories.shape)
    print("Shape of label tensor: ", labels.shape)
        
    # Randomize data
    np.random.shuffle(trainIndices)
    dataQueries = dataQueries[trainIndices]
    dataOwnHistories = dataOwnHistories[trainIndices]
    dataOtherHistories = dataOtherHistories[trainIndices]
    labels = labels[trainIndices]
    

    ## Calculating training batch sizes
    batches = zip(range(0, n_train, config["batch_size"]), range(config["batch_size"], n_train+config["batch_size"], config["batch_size"]))
    batches = [(start, end) for start, end in batches]

    model = train_model(config, dataQueries, dataOwnHistories, dataOtherHistories, labels, embeddingMatrix, batches)



    ####################  Test file generation  #############################


    print("Creating solution file...")

    # Preparing test data
    testQueries = pad_sequences(testQueriesSequences, maxlen=config["max_sequence_length"])
    testOwnHistories = datahelper.prepare_history(testOwnHistoriesSequences, mode="own", maxlen=config["max_sequence_length"])
    testOtherHistories = datahelper.prepare_history(testOtherHistoriesSequences, mode="other", maxlen=config["max_sequence_length"])

    ## Calculating testing batch sizes
    batches = zip(range(0, n_test, config["batch_size"]), range(config["batch_size"], n_test+config["batch_size"], config["batch_size"]))
    batches = [(start, end) for start, end in batches]

    predictions = predict_model(config, model, testQueries, testOwnHistories, testOtherHistories, batches)

    with io.open(config["solution_path"], "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(config["test_data_path"], encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(datahelper.label2emotion[predictions[lineNum]] + '\n')




if __name__ == '__main__':
    main()