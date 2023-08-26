# fits on 12GB

export PYTHONPATH='./' 
MASTER_PORT=${MASTER_PORT:-32001}

deepspeed \
    --master_port $MASTER_PORT \
    src/trainer/perlin_trainer.py \
    --model opt-125m \
    --dataset wikitext2 \
    --k 64 \
    --gradient-checkpointing \
    --predictor-length 256 \
    --performer-nb-feature-factor 8 \
    --deepspeed-enable \
    --deepspeed \
    --deepspeed_config ./config/ds_opt_125.json
    # --kd-checkpointing