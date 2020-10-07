# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

for SPLIT in train valid test; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "iemocap/$SPLIT.input0" \
        --outputs "iemocap/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done

# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  

fairseq-preprocess \
    --only-source \
    --trainpref "iemocap/train.input0.bpe" \
    --validpref "iemocap/valid.input0.bpe" \
    --destdir "iemocap-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "iemocap/train.label" \
    --validpref "iemocap/valid.label" \
    --destdir "iemocap-bin/label" \
    --workers 60