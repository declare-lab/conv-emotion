
# Models for Recognizing Emotion Cause in Conversations

This repository contains implementations of different architectures to detect emotion cause in conversations.

<p align="center">
  <img src="images/cause_types1.png" alt="Emotion cause types in conversation" width="1000"/>
  <figcaption stype="display:table-caption;"><em> (a) No context. (b) Unmentioned Latent Cause. (c) Distinguishing emotion cause from emotional expressions.</em></figcaption>
</p>

<p align="center">
  <img src="images/cause_types2.png" alt="Emotion cause types in conversation" width="1000"/>
  <figcaption stype="display:table-caption;"><em> (a) Self-contagion. (b) The cause of the emotion is primarily due to a stable mood of the speaker that was induced in the previous dialogue turns; (c) The hybrid type with both inter-personal emotional influence and self-contagion.</em></figcaption>
</p>

## Baseline Results on [RECCON](https://github.com/declare-lab/RECCON) dataset (DailyDialog Fold)

| Model 	| emo_f1 	| pos_f1 	| neg_f1 	| macro_avg 	|
|-	|-	|-	|-	|-	|
| ECPE-2d cross_road<br>(0 transform layer) 	| 52.76 	| 52.39 	| 95.86 	| 73.62 	|
| ECPE-2d window_constrained<br>(1 transform layer) 	| 70.48 	| 48.80 	| 93.85 	| 71.32 	|
| ECPE-2d cross_road<br>(2 transform layer) 	| 52.76 	| 55.50 	| 94.96 	| 75.23 	|
| ECPE-MLL | - | 48.48 | 94.68 | 71.58 |
| Rank Emotion Cause | - | 33.00 |  97.30 |  65.15 |
| RoBERTa-base | - | 64.28 |  88.74 |  76.51 |
| RoBERTa-large | - | 66.23 |  87.89 |  77.06 |

## ECPE-2D on [RECCON](https://github.com/declare-lab/RECCON) dataset

<p align="center">
  <img src="images/ECPE-2D.png" alt="ECPE-2D" width="1000"/>
</p>

Citation:
Please cite the following papers if you use this code.
- Recognizing Emotion Cause in Conversations. Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Romila Ghosh, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. Arxiv (2020). [[pdf](https://arxiv.org/pdf/2012.11820.pdf)]
- Zixiang Ding, Rui Xia, Jianfei Yu. ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.288.pdf)]

## Rank-Emotion-Cause on [RECCON](https://github.com/declare-lab/RECCON) dataset

<p align="center">
  <img src="images/rank-emotion-cause.png" alt="ECPE-2D" width="1000"/>
</p>

Citation:
Please cite the following papers if you use this code.
- Recognizing Emotion Cause in Conversations. Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Romila Ghosh, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. Arxiv (2020). [[pdf](https://arxiv.org/pdf/2012.11820.pdf)]
- **Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction**. In *Proc. of ACL 2020: The 58th Annual Meeting of the Association for Computational Linguistics*, pages 3171--3181. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.289/)] 


## ECPE-MLL on [RECCON](https://github.com/declare-lab/RECCON) dataset


<p align="center">
  <img src="images/ECPE-MLL.png" alt="ECPE-2D" width="1000"/>
</p>

Citation:
Please cite the following papers if you use this code.
- Recognizing Emotion Cause in Conversations. Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Romila Ghosh, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. Arxiv (2020). [[pdf](https://arxiv.org/pdf/2012.11820.pdf)]
- Zixiang Ding, Rui Xia, Jianfei Yu. End-to-End Emotion-Cause Pair Extraction based on SlidingWindow Multi-Label Learning. EMNLP 2020.[[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.290.pdf)]

## RoBERTa and SpanBERT Baselines on [RECCON](https://github.com/declare-lab/RECCON) dataset

The RoBERTa and SpanBERT baselines as explained in the original RECCON paper. Refer to [this](https://arxiv.org/pdf/2012.11820.pdf).

Citation:
Please cite the following papers if you use this code.
- Recognizing Emotion Cause in Conversations. Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Romila Ghosh, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. Arxiv (2020). [[pdf](https://arxiv.org/pdf/2012.11820.pdf)]
