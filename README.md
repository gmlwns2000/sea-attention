# SEA: Sparse Linear Attention with Estimated Attention Mask

Codebase for paper reproduce.

# How-to

```sh 
# to run opt variants using deepspeed and multi-gpus
# opt-125m and 350m are tuned for 11GB VRAM; opt-1.3b and opt-2.7b are tuned for 24GB VRAM;
./scripts/opt.py --help
# to run opt variants without deepspeed
python -m src.trainer.perlin_trainer --help
# to run bert variants
python -m src.trainer.perlin_trainer --help
```

- OPT
```sh
# famous env. vars
export MASTER_PORT=12311

# examples
./scripts/opt.py --method perlin --model opt-125m --k 64 --predictor-legnth 256 --nbf 8
./scripts/opt.py --method performer --model opt-125m --nbf 1
./scripts/opt.py --method reformer --model opt-125m --k 64

# load checkpoint
./scripts/opt.py --method perlin --model opt-125m --load-checkpoint auto

# without deepspeed
python -m src.trainer.perlin_trainer --dataset wikitext2 --method perlin --model opt-125m
```

- BERT
```sh
# examples
python -m src.trainer.perlin_trainer --dataset glue --subset mrpc --method perlin --k 64 --perdictor-length 128 --performer-nb-feature-factor 1
python -m src.trainer.perlin_trainer --dataset glue --subset mrpc --method performer --performer-nb-feature-factor 1
python -m src.trainer.perlin_trainer --dataset glue --subset mrpc --method reformer --k 64
python -m src.trainer.perlin_trainer --dataset glue --subset mrpc --method none

# evaluate only
python -m src.trainer.perlin_trainer --dataset glue --subset mrpc --method none --eval
```

- Development Test Codes

Located on `src.main.tests.*`.