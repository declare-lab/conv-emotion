TOTAL_NUM_UPDATES=7080  # 30 epochs through EmoryNLP for bsz 32
WARMUP_UPDATES=425      # 6 percent of the number of updates
LR=1e-5                 # Peak LR for polynomial LR scheduler.
HEAD_NAME="emorynlp_head"   # Custom name for the classification head.
NUM_CLASSES=7           # Number of classes for the classification task.
MAX_SENTENCES=8         # Batch size.
ROBERTA_PATH="roberta.large/model.pt"

CUDA_VISIBLE_DEVICES=0 python train.py "emorynlp-bin/" \
    --log-format "simple" \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 5000 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 30 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --find-unused-parameters \
    --shorten-method "truncate" \
    --update-freq 4 \
    --no-epoch-checkpoints \
    --save-dir "checkpoints/emorynlp/"