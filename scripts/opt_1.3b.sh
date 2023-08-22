# fits on 24GB

export PYTHONPATH='./' 
deepspeed src/trainer/perlin_trainer.py \
    --model opt-1.3b \
    --dataset wikitext2 \
    --k 64 \
    --gradient-checkpointing \
    --predictor-length 256 \
    --performer-nb-feature-factor 8 \
    --deepspeed-enable \
    --deepspeed \
    --deepspeed_config ./config/ds_opt_1.3.json
    # --kd-checkpointing