

## Setup

This code is adapted from [here](https://github.com/NUSTM/ECPE-MLL)

- **Python 3** (tested on python 3.6)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 2.4
- Bert (The pretrained bert model "[BERT-Base, English](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)" is required. The model is built based on
this implementation: https://github.com/google-research/bert)  Download the zip and put it under path `./BERT/BERT-base-english/`

```sh
python main.py
```
> Note that ECPE-2D only works on the task 2 i.e., Emotion Causal Entailment of the RECCON Dataset.

## Results
Emotion and Cause Pair Extraction result on RECCON([dailydialog_test](data_reccon/dailydialog_test.json)):

 |    Positive                       |  Negative                 |   Macro Average              |
 | :----------------------------------: | :--------------------------: | :--------------------------: |
 |   F=0.4848                          |  F=0.9468                      |   F=0.7158                          |
