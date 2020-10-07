The files in the `comet/` directory were almost entirely adopted from the original implementaion https://github.com/atcbosselut/comet-commonsense. 

To extract COMET commonsense features, first download the required files from [here](https://drive.google.com/file/d/1vNi4TViLKX_V_wGVXfhpvKimqMjhGBNX/view?usp=sharing). Keep the `atomic_pretrained_model.pickle` in `comet/pretrained_models/` and the other pickle file in `comet/data/atomic/processed/generation/`. Then you can extract commonesnse features for all the datasets using

```bash
python comet_feature_extract_all.py
```

To train the RoBERTa model, first download the pretrained weights from [here](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz) and uncompress the tar file so that the `roberta.large/` is placed in this directory. Then you can preprocess, train, and extract the context-independent feature vectros for the IEMOCAP dataset as follows:

```bash
python roberta_init_iemocap.py
bash roberta_preprocess_iemocap.sh
bash roberta_train_iemocap.sh
python roberta_feature_extract_iemocap.py
```

The features will be saved in the `iemocap/` directory.

Use the other corresponding scripts to extract the features for the other datasets. You can then use the COMET and RoBERTa features to train the models using the scripts in `COSMIC/erc-training`. For reproduction of our results, we provide the features that we used in our experimets [here](https://drive.google.com/file/d/1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr/view?usp=sharing).