# Emotion Recognition in Conversations

This repository contains implementations for three conversational emotion detection methods, namely:
- bc-LSTM (keras)
- CMN (tensorflow)
- ICON (tensorflow)
- DialogueRNN (PyTorch)

Unlike other emotion detection models, these techniques consider the party-states and inter-party dependencies for modeling conversational context relevant to emotion recognition. The primary purpose of all these techniques are to pretrain an emotion detection model for empathetic dialogue generation.

## Data Format

The networks expect emotion/sentiment label and speaker info for each utterance present in a dialogue. However, the code can be adpated to perform tasks where only the previous utterances are available without corresponding labels as context and goal is to label the present utterance. This code can also be trained end-to-end manner. We will soon push these desired changes. 

## bc-LSTM
[_bc-LSTM_](http://www.aclweb.org/anthology/P17-1081) is a network for using context to detection emotion of an utterance in a dialogue. The model is simple but efficient which only uses a LSTM to model the temporal relation among the utterances. In this repo we gave the data of Semeval 2019 Task 3. We have used and provided the data released by Semeval 2019 Task 3 - "Emotion Recognition in Context" organizers. In this task only 3 utterances have been provided - utterance1 (user1), utterance2 (user2), utterance3 (user1) consecutively. The task is to predict the emotion label of utterance3. Emotion label of each utterance have not been provided. However, if your data contains emotion label of each utterance then you can still use this code and adapt it accordingly. Hence, this code is still aplicable for the datasets like MOSI, MOSEI, IEMOCAP, AVEC, DailyDialogue etc. bc-LSTM does not make use of speaker information like CMN, ICON and DialogueRNN.

### Requirements

- python 3.6.5
- pandas==0.23.3
- tensorflow==1.9.0
- numpy==1.15.0
- scikit_learn==0.20.0
- keras==2.1

### Execution
1. `cd bc-LSTM`

2. Train the bc-LSTM model:
    - `python baseline.py -config testBaseline.config` for IEMOCAP

### Citation

Please cite the following paper if you find this code useful in your work.

```
Poria, S., Cambria, E., Hazarika, D., Majumder, N., Zadeh, A. and Morency, L.P., 2017. Context-dependent sentiment analysis in user-generated videos. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (Vol. 1, pp. 873-883).  
}
```
## CMN
[_CMN_](http://aclweb.org/anthology/N18-1193) is a neural framework for emotion detection in dyadic conversations. It leverages mutlimodal signals from text, audio and visual modalities. It specifically incorporates speaker-specific dependencies into its architecture for context modeling. Summaries are then generated from this context using multi-hop memory networks.

### Requirements

- python 3.6.5
- pandas==0.23.3
- tensorflow==1.9.0
- numpy==1.15.0
- scikit_learn==0.20.0

### Execution
1. `cd CMN`

2. Unzip the data as follows:  
    - Download the features for IEMOCAP using this [link](https://drive.google.com/file/d/1zWCN2oMdibFkOkgwMG2m02uZmSmynw8c/view?usp=sharing).
    - Unzip the folder and place it in the location: `/CMN/IEMOCAP/data/`. Sample command to achieve this: `unzip {path_to_zip_file} -d ./IEMOCAP/`
3. Train the ICON model:
    - `python train_iemocap.py` for IEMOCAP

### Citation

Please cite the following paper if you find this code useful in your work.

```
@inproceedings{hazarika2018conversational,  
    title={Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos},   
    author={Hazarika, Devamanyu and Poria, Soujanya and Zadeh, Amir and Cambria, Erik and Morency, Louis-Philippe and Zimmermann, Roger},   
    booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},   
    volume={1},   
    pages={2122--2132},   
    year={2018}   
}
```
## ICON

Interactive COnversational memory Network (ICON) is a multimodal emotion detection framework that extracts multimodal features from conversational videos and hierarchically models the \textit{self-} and \textit{inter-speaker} emotional influences into global memories. Such memories generate contextual summaries which aid in predicting the emotional orientation of utterance-videos.

### Requirements

- python 3.6.5
- pandas==0.23.3
- tensorflow==1.9.0
- numpy==1.15.0
- scikit_learn==0.20.0

### Execution
1. `cd ICON`

2. Unzip the data as follows:  
    - Download the features for IEMOCAP using this [link](https://drive.google.com/file/d/1zWCN2oMdibFkOkgwMG2m02uZmSmynw8c/view?usp=sharing).
    - Unzip the folder and place it in the location: `/ICON/IEMOCAP/data/`. Sample command to achieve this: `unzip  {path_to_zip_file} -d ./IEMOCAP/`
3. Train the ICON model:
    - `python train_iemocap.py` for IEMOCAP

### Citation
`ICON: Interactive Conversational Memory Networkfor Multimodal Emotion Detection. D. Hazarika, S. Poria, R. Mihalcea, E. Cambria, and R. Zimmermann. EMNLP (2018), Brussels, Belgium`

## DialogueRNN: An Attentive RNN for Emotion Detection in Conversations

[_DialogueRNN_](https://arxiv.org/pdf/1811.00405.pdf) is basically a customized recurrent neural network (RNN) that
profiles each speaker in a conversation/dialogue on the fly, while models the
context of the conversation at the same time. This model can easily be extended to
multi-party scenario. Also, it can be used as a pretraining model for empathetic
dialogue generation.

### Requirements

- Python 3
- PyTorch 0.4
- Pandas 0.23
- Scikit-Learn 0.20
- TensorFlow (optional; required for tensorboard)
- tensorboardX (optional; required for tensorboard)

### Execution

1. _IEMOCAP_ dataset: `python train_IEMOCAP.py`
2. _AVEC_ dataset: `python train_AVEC.py`

### Dataset Features

Please extract the file `DialogueRNN_features.zip`.

### Command-Line Arguments

-  `--no-cuda`: Does not use GPU
-  `--lr`: Learning rate
-  `--l2`: L2 regularization weight
-  `--rec-dropout`: Recurrent dropout
-  `--dropout`: Dropout
-  `--batch-size`: Batch size
-  `--epochs`: Number of epochs
-  `--class-weight`: class weight (not applicable for AVEC)
-  `--active-listener`: Explicit lisnener mode
-  `--attention`: Attention type
-  `--tensorboard`: Enables tensorboard log

### Citation

Please cite the following paper if you find this code useful in your work.

`DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, E. Cambria, and G. Alexander. AAAI (2019), Honolulu, Hawaii, USA`
