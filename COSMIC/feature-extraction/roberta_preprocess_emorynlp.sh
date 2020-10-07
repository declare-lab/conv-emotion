for SPLIT in train valid test; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "emorynlp/$SPLIT.input0" \
        --outputs "emorynlp/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done 

fairseq-preprocess \
    --only-source \
    --trainpref "emorynlp/train.input0.bpe" \
    --validpref "emorynlp/valid.input0.bpe" \
    --destdir "emorynlp-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "emorynlp/train.label" \
    --validpref "emorynlp/valid.label" \
    --destdir "emorynlp-bin/label" \
    --workers 60