# Emotion Recognition in Conversations

This repository contains implementations for three conversational emotion detection methods, namely:
- CMN (tensorflow)
- ICON (tensorflow)
- DialogueRNN (PyTorch)

## CMN

## ICON

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

1. _IEMOCAP_ dataset: train_IEMOCAP.py
2. _AVEC_ dataset: train_AVEC.py

### Dataset Feature

Please use this
[link](https://drive.google.com/file/d/19-KVno1Ki63h78hm48FjJQHKSZV84rPo/view?usp=sharing)
to get the features for IEMOCAP and AVEC dataset.

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

Please cite the paper `arXiv:1811.00405` if you find this code useful in your work.
