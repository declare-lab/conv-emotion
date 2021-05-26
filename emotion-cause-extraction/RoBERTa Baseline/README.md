# RECCON: Recognizing Emotion Cause in CONversations

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recognizing-emotion-cause-in-conversations-1/causal-emotion-entailment-on-reccon)](https://paperswithcode.com/sota/causal-emotion-entailment-on-reccon?p=recognizing-emotion-cause-in-conversations-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recognizing-emotion-cause-in-conversations-1/recognizing-emotion-cause-in-conversations-on)](https://paperswithcode.com/sota/recognizing-emotion-cause-in-conversations-on?p=recognizing-emotion-cause-in-conversations-1)

This repository contains the dataset and the pytorch implementations of the models from the paper [Recognizing Emotion Cause in Conversations](https://arxiv.org/pdf/2012.11820.pdf).

## Overview of the Task

![Alt text](RECCON.png?raw=true "Task Details")

Given an utterance U, labeled with emotion E, the task is to extract the causal spans S from the conversational history H (including utterance U) that sufficiently represent the causes of emotion E.

## Dataset

The original annotated dataset can be found in the json files in the `data/original_annotation` folder. The dataset with negative examples for the Causal Span Extraction and the Causal Entailment of Emotion tasks can be found in `data/subtask1/` and `data/subtask2/` folders respectively.

### Data Format

The annotations and dialogues of the DailyDialog and IEMOCAP are available at [`data/original_annotation/*.json`](data/original_annotation/).
Each instance in the JSON file is allotted one identifier (e.g. "tr\_10180") which is a list having a dictionary of the following items for each utterance:   

| Key                                | Value                                                                        | 
| ---------------------------------- |:----------------------------------------------------------------------------:| 
| `turn`                             | Utterance index starting from 1.                                             |
| `speaker`                          | Speaker of the target utterance.                                             | 
| `utterance`                        | The text of the utterance.                                                   | 
| `emotion`                          | Emotion label of the utterance.                                              | 
| `expanded emotion cause evidence`  | Utterance indices indicating the cause of a non neutral target utterance.    | 
| `expanded emotion cause spans`     | Causal spans corresponding to the evidence utterances.                       |
| `explanation`                      | Only if the annotator wrote any explanation about the emotion cause.         | 
| `type`                             | The type of the emotion cause.                                               | 


Example format in JSON:

```json
{
  "tr_10180": 
  [
    [
        {
            "turn": 1,
            "speaker": "A",
            "utterance": "It's time for desserts ! Are you still hungry ?",
            "emotion": "neutral"
        },
        {
            "turn": 2,
            "speaker": "B",
            "utterance": "I've always got room for something sweet !",
            "emotion": "happiness",
            "expanded emotion cause evidence": [
                1,
                2
            ],
            "expanded emotion cause span": [
                "desserts",
                "I've always got room for something sweet !"
            ],
            "type": [
                "no-context",
                "inter-personal"
            ]
        }

    ]
  ]
}
```

## Causal Span Extraction

We formulate the Causal Span Extraction task as a question answering task. To train RoBERTa or SpanBERT models for this task on the DailyDialog dataset use the following command:

`python train_qa.py --model [rob|span] --fold [1|2|3] --context`

Then, evlaution can be carried out on DailyDialog or IEMOCAP as follows:

`python eval_qa.py --model [rob|span] --fold [1|2|3] --context --dataset [dailydialog|iemocap]`


## Causal Entailment of Emotion

The Causal Entailment of Emotion task is formulated as a classification task. To train RoBERTa-Base or RoBERTa-Large models for this task on the DailyDialog dataset use the following command:

`python train_classification.py --model [rob|robl] --fold [1|2|3] --context`

Then, evlaution can be carried out on DailyDialog or IEMOCAP as follows:

`python eval_classification.py --model [rob|robl] --fold [1|2|3] --context --dataset [dailydialog|iemocap]`

Without context models can be trained and evaluated by removing `--context` from the above commands.

## Citation

Recognizing Emotion Cause in Conversations. Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Romila Ghosh, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. Arxiv (2020).
