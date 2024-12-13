TASK=${MODE:-MultiRC}
MODEL=${MODEL:-/data/peft_finetuning/models/opt-6.7b} #/data/hybrid-opt-FT/opt-13b
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-5}
BS=${BS:-2}
LR=${LR:-1e-6}
SEED=${SEED:-1}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
NUM_GPU=${NUM_GPU:-8}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=fsdp-$MODE-$EPOCH-$BS-$LR-$SEED

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
echo "EPOCH: $EPOCH"
echo "BS (gradient accumulation): $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

OMP_NUM_THREADS=10 torchrun --nproc_per_node=$NUM_GPU --master_port=$(( RANDOM + 1000 )) run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --trainer regular --fp16 --no_auto_device \
    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS --gradient_accumulation_steps 1 \
    --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 \
    --train_as_classification \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" > logs/$TASK-$MODEL_NAME-$MODE-$TAG-$SEED-8gpu.txt  2>&1
