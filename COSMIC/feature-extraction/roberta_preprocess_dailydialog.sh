for SPLIT in train valid test; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "dailydialog/$SPLIT.input0" \
        --outputs "dailydialog/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done 

fairseq-preprocess \
    --only-source \
    --trainpref "dailydialog/train.input0.bpe" \
    --validpref "dailydialog/valid.input0.bpe" \
    --destdir "dailydialog-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "dailydialog/train.label" \
    --validpref "dailydialog/valid.label" \
    --destdir "dailydialog-bin/label" \
    --workers 60