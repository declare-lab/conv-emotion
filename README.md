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

Please cite the paper `N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, E. Cambria, and G. Alexander. DialogueRNN: An
Attentive RNN for Emotion Detection in Conversations". In: AAAI. Vol. 1. 2019.` if you find this code useful in your work.
