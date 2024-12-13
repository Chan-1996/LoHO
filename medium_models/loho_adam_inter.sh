#!/bin/bash
TASK=${TASK:-MNLI}
K=${K:-512}
SEED=${SEED:-42}
BS=${BS:-64}
FO_LR=${LR:-5e-5}
ZO_LR=${LR:-1e-12}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-roberta-large}
MODEL_NAME=${MODEL_NAME:-"roberta-large"}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "ZO_LR: $ZO_LR"
echo "FO_LR: $FO_LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"

GR_TAG=seed$SEED-bs$BS-lr$ZO_LR-$FO_LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODEL_NAME}-mezo-adam-last-8layer-linear-fo-lr-no-dropout-${EXTRA_TAG}-test}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"
TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
      bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $FO_LR --zo_learning_rate $ZO_LR\
      --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
      --zo_combine_fo --fo_optim_layers "layer.12." "layer.13." "layer.14." "layer.15." "layer.16." "layer.17." "layer.18." "layer.19." "layer.20." "layer.21." "layer.22." "layer.23." \
      --lr_scheduler_type "linear" --optimizer "adam" --efficient_zero_order \
      $@