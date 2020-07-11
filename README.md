# Emotion Recognition in Conversations

### Updates

| 11/07/2020  | Interested in reading the papers on ERC or related tasks such as sarcasm detection in conversations? We have compiled a comprehensive reading list for papers. Please visit https://github.com/declare-lab/awesome-erc |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 07/06/2020: | New state-of-the-art results for the ERC task will be released soon.                                                                                                                                                   |
| 07/06/2020: | The conv-emotion repo will be maintained on https://github.com/declare-lab/                                                                                                                                            |
| 22/12/2019: | Code for DialogueGCN has been released.                                                                                                                                                                                |
| 11/10/2019: | New Paper: Conversational Transfer Learning for Emotion Recognition.                                                                                                                                                   |
| 09/08/2019: | New paper on Emotion Recognition in Conversation (ERC).                                                                                                                                                                |
| 06/03/2019: | Features and codes to train DialogueRNN on the MELD dataset have been released.                                                                                                                                        |
| 20/11/2018: | End-to-end version of ICON and DialogueRNN have been released.                                                                                                                                                         |
---------------------------------------------------------------------------

This repository contains implementations for three conversational emotion detection methods, namely:

- [Emotion Recognition in Conversations](#emotion-recognition-in-conversations)
  * [Data Format](#data-format)
  * [TL-ERC (PyTorch)](#tl-erc-emotion-recognition-in-conversations-with-transfer-learning-from-generative-conversation-modeling)
  * [DialogueGCN (PyTorch)](#dialoguegcn-a-graph-convolutional-neural-network-for-emotion-recognition-in-conversation)
  * [DialogueRNN (PyTorch)](#dialoguernn-an-attentive-rnn-for-emotion-detection-in-conversations)
  * [ICON (tensorflow)](#icon)
  * [CMN (tensorflow)](#cmn)
  * [bc-LSTM-pytorch (PyTorch)](#bc-lstm-pytorch)
  * [bc-LSTM (keras)](#bc-lstm)


Unlike other emotion detection models, these techniques consider the party-states and inter-party dependencies for modeling conversational context relevant to emotion recognition. The primary purpose of all these techniques are to pretrain an emotion detection model for empathetic dialogue generation.

<p align="center">
  <img src="bayesian_dialogue.jpg" alt="Controlling variables in conversation" width="600"/>
  <figcaption stype="display:table-caption;"><em>Fig. 1: Interaction among different controlling variables during a
dyadic conversation between persons A and B. Grey and white circles
represent hidden and observed variables, respectively. P represents
personality, U represents utterance, S represents interlocutor state, I
represents interlocutor intent, E represents emotion and Topic represents
topic of the conversation. This can easily be extended to multi-party
      conversations.</em></figcaption>
</p>



Emotion recognition can be very useful for empathetic and affective dialogue generation - 

<p align="center">
  <img src="affective_dialogue.jpg" alt="Affective dialogue generation" width="500"/>
</p>


## Data Format

These networks expect emotion/sentiment label and speaker info for each utterance present in a dialogue like
```
Party 1: I hate my girlfriend (angry)
Party 2: you got a girlfriend?! (surprise)
Party 1: yes (angry)
```
However, the code can be adpated to perform tasks where only the preceding utterances are available, without their corresponding labels, as context and goal is to label only the present/target utterance. For example, the *context* is
```
Party 1: I hate my girlfriend
Party 2: you got a girlfriend?!
```
the *target* is
```
Party 1: yes (angry)
```
where the target emotion is _angry_.
Moreover, this code can also be molded to train the network in an end-to-end manner. We will soon push these useful changes. 

## TL-ERC: Emotion Recognition in Conversations with Transfer Learning from Generative Conversation Modeling

[_TL-ERC_](https://arxiv.org/pdf/1910.04980.pdf) is a transfer learning-based framework for ERC. It pre-trains a generative dialogue model and transfers context-level weights that include affective knowledge into the target discriminative model for ERC.

<p align="center">
  <img src="TL-ERC.jpg" alt="TL-ERC framework" width="600"/>
</p>

### Setting up

1. Setup an environment with Conda:

    ```bash
    conda env create -f environment.yml
    conda activate TL_ERC
    cd TL_ERC
    python setup.py
    ```
2. Download dataset files [IEMOCAP](https://drive.google.com/file/d/1nufbrBJ-LtcROv1MviCHFI7tQE3JnqQR/view?usp=sharing), [DailyDialog](https://drive.google.com/file/d/13rHLtAMmDsiCP1hZwWqMNcrtkmZF_hK3/view?usp=sharing) and store them in `./datasets/`.

3. Download the pre-trained weights of HRED on [Cornell](https://drive.google.com/file/d/1OXtnyJ5nDMmK75L9kEQvKPIyO0xzyeVC/view?usp=sharing) and [Ubuntu](https://drive.google.com/file/d/1T2HLfSvWr7CSrhBuE193XRRXwfLkO_aK/view?usp=sharing) datasets and store them in `./generative_weights/`

4. [Optional]: To train new generative weights from dialogue models, refer to https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling . 


### Run the ERC classifier with pre-trained weights

1. `cd bert_model`
2. `python train.py --load_checkpoint=../generative_weights/cornell_weights.pkl --data=iemocap`.   
    -  Change `cornell` to `ubuntu` and `iemocap` to `dailydialog` for other dataset combinations.
    -  Drop `load_checkpoint` to avoid initializing contextual weights.
    -  To modify hyperparameters, check `configs.py`

### [Optional] Create ERC Dataset splits

1. Set [glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) path in the preprocessing files.
2. `python iemocap_preprocess.py`. Similarly for `dailydialog`.

### Citation

Please cite the following paper if you find this code useful in your work.

```bash
@article{hazarika2020conversational,
  title={Conversational Transfer Learning for Emotion Recognition},
  author={Hazarika, Devamanyu and Poria, Soujanya and Zimmermann, Roger and Mihalcea, Rada},
  journal={Information Fusion},
  year={2020}
}
```
## DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation

[_DialogueGCN_](https://arxiv.org/pdf/1908.11540.pdf) (Dialogue Graph Convolutional Network), is a graph neural network based approach to ERC. We leverage self and inter-speaker dependency of the interlocutors to model conversational context for emotion recognition. Through the graph network, DialogueGCN addresses context propagation issues present in the current RNN-based methods. DialogueGCN is naturally suited for multi-party dialogues.

![Alt text](DialogueGCN.jpg?raw=true "DialogueGCN framework")
### Requirements

- Python 3
- PyTorch 1.0
- PyTorch Geometric 1.3
- Pandas 0.23
- Scikit-Learn 0.20
- TensorFlow (optional; required for tensorboard)
- tensorboardX (optional; required for tensorboard)

### Execution

__Note__: PyTorch Geometric makes heavy usage of CUDA atomic operations and is a source of non-determinism. To reproduce the results reported in the paper, we recommend to use the following execution command. Note that this script will execute in CPU. We obatined weighted average F1 scores of 64.67 in our machine and 64.44 in Google colaboratory for IEMOCAP dataset with the following command.

1. _IEMOCAP_ dataset: `python train_IEMOCAP.py --base-model 'LSTM' --graph-model --nodal-attention --dropout 0.4 --lr 0.0003 --batch-size 32 --class-weight --l2 0.0 --no-cuda`

### Citation

Please cite the following paper if you find this code useful in your work.

`DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. D. Ghosal, N. Majumder, S. Poria, N. Chhaya, & A. Gelbukh. EMNLP-IJCNLP (2019), Hong Kong, China.`


## DialogueRNN: An Attentive RNN for Emotion Detection in Conversations

[_DialogueRNN_](https://arxiv.org/pdf/1811.00405.pdf) is basically a customized recurrent neural network (RNN) that
profiles each speaker in a conversation/dialogue on the fly, while models the
context of the conversation at the same time. This model can easily be extended to
multi-party scenario. Also, it can be used as a pretraining model for empathetic
dialogue generation. 

__Note__: the default settings (hyperparameters and commandline arguments) in the code are meant for BiDialogueRNN+Att. The user needs to optimize the settings for other the variants and changes.
![Alt text](dialoguernn.jpg?raw=true "DialogueRNN framework")
### Requirements

- Python 3
- PyTorch 1.0
- Pandas 0.23
- Scikit-Learn 0.20
- TensorFlow (optional; required for tensorboard)
- tensorboardX (optional; required for tensorboard)

### Dataset Features

Please extract the file `DialogueRNN_features.zip`.

### Execution

1. _IEMOCAP_ dataset: `python train_IEMOCAP.py <command-line arguments>`
2. _AVEC_ dataset: `python train_AVEC.py <command-line arguments>`

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
-  `--attribute`: Attribute 1 to 4 (only for AVEC)

### Citation

Please cite the following paper if you find this code useful in your work.

`DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, E. Cambria, and G. Alexander. AAAI (2019), Honolulu, Hawaii, USA`

## ICON

Interactive COnversational memory Network (ICON) is a multimodal emotion detection framework that extracts multimodal features from conversational videos and hierarchically models the \textit{self-} and \textit{inter-speaker} emotional influences into global memories. Such memories generate contextual summaries which aid in predicting the emotional orientation of utterance-videos.
<p align="center">
  <img src="icon.jpg" alt="ICON framework" width="400"/>
</p>

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

## CMN
[_CMN_](http://aclweb.org/anthology/N18-1193) is a neural framework for emotion detection in dyadic conversations. It leverages mutlimodal signals from text, audio and visual modalities. It specifically incorporates speaker-specific dependencies into its architecture for context modeling. Summaries are then generated from this context using multi-hop memory networks.
![Alt text](cmn.jpg?raw=true "CMN framework")

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
Hazarika, D., Poria, S., Zadeh, A., Cambria, E., Morency, L.P. and Zimmermann, R., 2018. Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (Vol. 1, pp. 2122-2132).
```

## bc-LSTM-pytorch
[_bc-LSTM-pytorch_](http://www.aclweb.org/anthology/P17-1081) is a network for using context to detection emotion of an utterance in a dialogue. The model is simple but efficient which only uses a LSTM to model the temporal relation among the utterances. In this repo we gave the data of Semeval 2019 Task 3. We have used and provided the data released by Semeval 2019 Task 3 - "Emotion Recognition in Context" organizers. In this task only 3 utterances have been provided - utterance1 (user1), utterance2 (user2), utterance3 (user1) consecutively. The task is to predict the emotion label of utterance3. Emotion label of each utterance have not been provided. However, if your data contains emotion label of each utterance then you can still use this code and adapt it accordingly. Hence, this code is still aplicable for the datasets like MOSI, MOSEI, IEMOCAP, AVEC, DailyDialogue etc. bc-LSTM does not make use of speaker information like CMN, ICON and DialogueRNN.
<p align="center">
  <img src="bclstm.jpg" alt="bc-LSTM framework" width="500"/>
</p>

### Requirements

- python 3.6.5
- pandas==0.23.3
- PyTorch 1.0
- numpy==1.15.0
- scikit_learn==0.20.0

### Execution
1. `cd bc-LSTM-pytorch`

2. Train the bc-LSTM model:
    - `python train_IEMOCAP.py` for IEMOCAP

### Citation

Please cite the following paper if you find this code useful in your work.

```
Poria, S., Cambria, E., Hazarika, D., Majumder, N., Zadeh, A. and Morency, L.P., 2017. Context-dependent sentiment analysis in user-generated videos. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (Vol. 1, pp. 873-883).
```

## bc-LSTM
[_bc-LSTM_](http://www.aclweb.org/anthology/P17-1081) is a network for using context to detection emotion of an utterance in a dialogue. The model is simple but efficient which only uses a LSTM to model the temporal relation among the utterances. In this repo we gave the data of Semeval 2019 Task 3. We have used and provided the data released by Semeval 2019 Task 3 - "Emotion Recognition in Context" organizers. In this task only 3 utterances have been provided - utterance1 (user1), utterance2 (user2), utterance3 (user1) consecutively. The task is to predict the emotion label of utterance3. Emotion label of each utterance have not been provided. However, if your data contains emotion label of each utterance then you can still use this code and adapt it accordingly. Hence, this code is still aplicable for the datasets like MOSI, MOSEI, IEMOCAP, AVEC, DailyDialogue etc. bc-LSTM does not make use of speaker information like CMN, ICON and DialogueRNN.
<p align="center">
  <img src="bclstm.jpg" alt="bc-LSTM framework" width="500"/>
</p>

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
```

