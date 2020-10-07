for SPLIT in train valid test; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "meld/$SPLIT.input0" \
        --outputs "meld/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done 

fairseq-preprocess \
    --only-source \
    --trainpref "meld/train.input0.bpe" \
    --validpref "meld/valid.input0.bpe" \
    --destdir "meld-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "meld/train.label" \
    --validpref "meld/valid.label" \
    --destdir "meld-bin/label" \
    --workers 60