TASK=${MODE:-XSum}
MODEL=${MODEL:-/data/hybrid-opt-FT/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-1e-7}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-10000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000} #20000
EVAL_STEPS=${EVAL_STEPS:-2000} #4000

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

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
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

CUDA_VISIBLE_DEVICES=2 python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS  --max_length 512 --max_new_tokens 64 \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" > logs/$TASK-$MODEL_NAME-$MODE-$TAG-$SEED.txt  2>&1
