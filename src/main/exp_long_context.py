import copy
import itertools
import os, sys, subprocess, json
import time
from .benchmark_opt_ablation import exam_config, BenchConfig

"""
__LOAD_PREFIX=dev5
DYNAMIC_K=128
QUERY_SKIPS=1
LOAD_AFTER_RESIZE=1
__CONTEXT=4096 
__STRIDE=4096 
python -m src.trainer.perlin_trainer \
    --model opt-125m \
    --method perlin \
    --dataset wikitext2 \
    --k 128 \
    --predictor-length 96 \
    --performer-nb-feature-factor 8.0 \
    --context-output-method mix \
    --load-checkpoint auto \
    --eval
"""

load_prefix = 'dev5'
k = 128
pw = 96

dynamic_ks = [96, 104, 112, 120, 128]
query_skips = [1, 2, 4, 8, 16]

def long_sleep(sec):
    last_tick = time.time()
    elapsed = 0
    while elapsed < sec:
        time.sleep(0.1)
        elapsed += time.time() - last_tick
        last_tick = time.time()
        print(f'sleeping ({elapsed:.1f} sec)\r', end='', flush=True)
    print()

def samples():
    options = itertools.product(query_skips, dynamic_ks)
    data = {}
    for qskip, dks in options:
        envs = copy.deepcopy(os.environ)
        envs.update({
            '__LOAD_PREFIX': load_prefix,
            'DYNAMIC_K': str(int(dks)),
            'QUERY_SKIPS': str(int(qskip)),
            'LOAD_AFTER_RESIZE': '1',
            '__CONTEXT': '4096',
            '__STRIDE': '4096',
            # 'CUDA_VISIBLE_DEVICES': '0',
        })
        cmd = \
            f'python -m src.trainer.perlin_trainer '\
            f'--model opt-125m '\
            f'--method perlin '\
            f'--dataset wikitext2 '\
            f'--k {int(k)} '\
            f'--predictor-length {int(pw)} '\
            f'--performer-nb-feature-factor 8.0 '\
            f'--context-output-method mix '\
            f'--load-checkpoint auto '\
            f'--eval'
        subprocess.call(cmd.split(' '), env=envs)
        with open('./cache/perlin_trainer/last_ppl.txt', 'r') as f:
            text = f.read()
            text = text.strip().replace('\n', '')
            ppl = float(text)
        os.environ.update({
            'DYNAMIC_K': str(int(dks)),
            'QUERY_SKIPS': str(int(qskip)),
        })
        long_sleep(60)
        latency, mem = exam_config(BenchConfig(
            method='perlin',
            seq_len=4096,
            k=k,
            w=pw,
            trace=False,
            causal=True
        ))
        # long_sleep(120)
        latency = latency * 1000
        mem = mem / (1024 ** 2)
        sample = {
            'mem': mem,
            'latency': latency,
            'query_skip': qskip,
            'dynamic_k': dks,
            'ppl': ppl,
        }
        print(sample)
        data[f'{dks},{qskip}'] = sample
    
    os.makedirs('./plots/exp_long_context', exist_ok=True)
    with open('./plots/exp_long_context/data.json', 'w') as f:
        json.dump(data, f, indent=2)

def main():
    samples()
    
if __name__ == '__main__':
    main()