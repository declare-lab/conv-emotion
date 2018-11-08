import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.models import load_model
from pathlib import Path
import json, argparse, os
import re, pickle
import io
import sys
import os





class DataHelper:

    def __init__(self, config):
        self.config = config

        # Label mappings
        self.label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
        self.emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

    def preprocessData(self, dataFilePath, mode):
        """Load data from a file, process and return indices, conversations and labels in separate lists
        Input:
            dataFilePath : Path to train/test file to be processed
            mode : "train" mode returns labels. "test" mode doesn't return labels.
        Output:
            indices : Unique conversation ID list
            queries : List of turn 3 sentences
            ownHistories : List of turn 1 sentences
            otherHistories: List of turn 2 sentences
            conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
            labels : [Only available in "train" mode] List of labels
        """
        indices = []
        conversations = []
        queries, ownHistories, otherHistories = [], [], []
        conversations = []
        labels = []
        with io.open(dataFilePath, encoding="utf8") as finput:
            finput.readline()
            for line in finput:
                # Convert multiple instances of . ? ! , to single instance
                # okay...sure -> okay . sure
                # okay???sure -> okay ? sure
                # Add whitespace around such punctuation
                # okay!sure -> okay ! sure
                repeatedChars = ['.', '?', '!', ',']
                for c in repeatedChars:
                    lineSplit = line.split(c)
                    while True:
                        try:
                            lineSplit.remove('')
                        except:
                            break
                    cSpace = ' ' + c + ' '    
                    line = cSpace.join(lineSplit)
                
                line = line.strip().split('\t')
                if mode == "train":
                    # Train data contains id, 3 turns and label
                    label = self.emotion2label[line[4]]
                    labels.append(label)

                ownHistory = line[1]
                otherHistory = line[2]
                query = line[3]
                conv = ' <eos> '.join(line[1:4])

                # Remove any duplicate spaces
                duplicateSpacePattern = re.compile(r'\ +')
                ownHistory = re.sub(duplicateSpacePattern, ' ', ownHistory)
                otherHistory = re.sub(duplicateSpacePattern, ' ', otherHistory)
                query = re.sub(duplicateSpacePattern, ' ', query)
                conv = re.sub(duplicateSpacePattern, ' ', conv)
                
                indices.append(int(line[0]))
                queries.append(query.lower())
                ownHistories.append(ownHistory.lower())
                otherHistories.append(otherHistory.lower())
                conversations.append(conv.lower())
        
        if mode == "train":
            return indices, queries, ownHistories, otherHistories, conversations, labels
        else:
            return indices, queries, ownHistories, otherHistories, conversations


    def getMetrics(self, predictions, ground):
        """Given predicted labels and the respective ground truth labels, display some metrics
        Input: shape [# of samples, NUM_CLASSES]
            predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
            ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
        Output:
            accuracy : Average accuracy
            microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
            microRecall : Recall calculated on a micro level
            microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
        """
        # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
        discretePredictions = to_categorical(predictions.argmax(axis=1))
        
        truePositives = np.sum(discretePredictions*ground, axis=0)
        falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
        falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
        
        print("True Positives per class : ", truePositives)
        print("False Positives per class : ", falsePositives)
        print("False Negatives per class : ", falseNegatives)
        
        # ------------- Macro level calculation ---------------
        macroPrecision = 0
        macroRecall = 0
        # We ignore the "Others" class during the calculation of Precision, Recall and F1
        for c in range(1, NUM_CLASSES):
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
        
        predictions = predictions.argmax(axis=1)
        ground = ground.argmax(axis=1)
        accuracy = np.mean(predictions==ground)
        
        print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
        return accuracy, microPrecision, microRecall, microF1


    def getEmbeddingMatrix(self, wordIndex):
        """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
           the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
        Input:
            wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
        Output:
            embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
        """

        embeddings_file = Path("./tmp/embeddings.p")

        if not embeddings_file.is_file():
            embeddingsIndex = {}
            # Load the embedding vectors from ther GloVe file
            with io.open(os.path.join(self.config["glove_dir"], 'glove.840B.300d.txt'), encoding="utf8") as f:
                for line in f:
                    try:
                        values = line.split()
                        word = values[0]
                        embeddingVector = np.asarray(values[1:], dtype='float32')
                        embeddingsIndex[word] = embeddingVector
                    except:
                        continue

            print('Glove word vectors : %s' % len(embeddingsIndex))
            
            # Minimum word index of any word is 1. 
            embeddingMatrix = np.zeros((len(wordIndex) + 1, self.config["embedding_dim"]))
            tokens_found=0
            for word, i in wordIndex.items():
                embeddingVector = embeddingsIndex.get(word)
                if embeddingVector is not None:
                    tokens_found+=1
                    # words not found in embedding index will be all-zeros.
                    embeddingMatrix[i] = embeddingVector

            print('Matching word vectors : %s' % len(embeddingsIndex))
            print('Tokens matched: {}/{}'.format(tokens_found, len(wordIndex)))

            pickle.dump( [wordIndex, embeddingsIndex, embeddingMatrix], open("./tmp/embeddings.p", "wb"))
        else:
            wordIndex, embeddingsIndex, embeddingMatrix = pickle.load(open("./tmp/embeddings.p", "rb"))
        
        return embeddingMatrix


    def prepare_history(self, data, mode, maxlen):
        data = pad_sequences(data, maxlen) # (batch, maxlen)
        pads = np.zeros(data.shape, dtype=np.float32) # (batch, maxlen)
        if mode == "own":
            data = np.stack((data, pads), axis=1)
        else:
            data = np.stack((pads, data), axis=1)
        return data # (batch, 2, maxlen)
