## Setup

This code is adapted from [Determined22/Rank-Emotion-Cause](https://github.com/Determined22/Rank-Emotion-Cause/)
- Python 3
- PyTorch 1.2.0
- [Transformers from Hugging Face](https://github.com/huggingface/transformers)

With Anaconda, we can create the environment with the provided `environment.yml`:

```bash
conda env create --file environment.yml 
conda activate EmoCau
```
To run the model
```bash
python src/main.py
```
> Note that rank-emotion-cause only works on the task 2 i.e., Emotion Causal Entailment of the RECCON Dataset.

## Results
Emotion and Cause Pair Extraction result on RECCON([dailydialog_test](data_reccon/dailydialog_test.json)):

 |    Positive                       |  Negative                 |   Macro Average              |
 | :----------------------------------: | :--------------------------: | :--------------------------: |
 |   F=0.33                         |  F=0.973                     |   F=0.6515                           |

<br>

