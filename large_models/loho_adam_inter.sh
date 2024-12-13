TASK=${MODE:-WIC}
MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
FO_LR=${LR:-5e-7}
ZO_LR=${LR:-1e-12}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-100}

MODE=${MODE:-ft-with-hybrid-opt}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

TAG=mezo-$MODE-last-20layer-adam-step-1-no-dropout-$STEPS-$BS-$FO_LR-$ZO_LR-$EPS-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "FO_LR: $FO_LR"
echo "ZO_LR: $ZO_LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

CUDA_VISIBLE_DEVICES=0 python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 1 \
    --max_steps $STEPS \
    --trainer zo --load_float16 --fo_optimizer "adam" --layer_wise_hybrid \
    --hybrid_optimizer True --sgd_optim_step 1 --alternate_training False --if_sgd_dropout False \
    --sgd_optim_layers "layers.20." "layers.21." "layers.22." "layers.23." "layers.24." "layers.25." "layers.26." "layers.27." "layers.28." "layers.29." "layers.30." "layers.31." "layers.32." "layers.33." "layers.34." "layers.35." "layers.36." "layers.37." "layers.38." "layers.39." \
    --zo_learning_rate $ZO_LR --learning_rate $FO_LR  --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS  \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --lr_scheduler_type "linear" --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" > logs/$TASK-$MODEL_NAME-$TAG-$SEED.txt  2>&1
