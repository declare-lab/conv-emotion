## Setup

This code is adapted from [here](https://github.com/NUSTM/ECPE-2D)

- **Python 3** (tested on python 3.6)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 2.4
- Bert (The pretrained bert model "[BERT-Base, English](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)" is required. The model is built based on
this implementation: https://github.com/google-research/bert)  Download the zip and put it under path `./BERT/BERT-base-english/`

## Usage
```sh
python main.py
```
> Note that ECPE-2D only works on the task 2 i.e., Emotion Causal Entailment of the RECCON Dataset.

## Results
Emotion and Cause Pair Extraction result on RECCON([dailydialog_test](data_reccon/dailydialog_test.json)):
| Model 	| emo_f1 	| pos_f1 	| neg_f1 	| macro_avg 	|
|-	|-	|-	|-	|-	|
| cross_road<br>(0 transform layer) 	| 52.76 	| 52.39 	| 95.86 	| 73.62 	|
| window_constrained<br>(1 transform layer) 	| 70.48 	| 48.80 	| 93.85 	| 71.32 	|
| cross_road<br>(2 transform layer) 	| 52.76 	| 55.50 	| 94.96 	| 75.23 	|

