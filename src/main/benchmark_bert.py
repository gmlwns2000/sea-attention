from dataclasses import dataclass
from matplotlib import pyplot as plt
import torch.multiprocessing as mp
import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random, json
from transformers import AutoConfig
from ..models.hf_bert import BertLayer
from ..models.hf_bert import BertModel as TeacherBertModel
from ..models import perlin_bert
from ..models.perlin_bert import BertModel, BertSelfAttention
from ..models.perlin_attention.config import PerlinAttentionConfig, register_default_config
from ..utils import seed, get_bench
from torch import nn
import json

plt.style.use('seaborn-bright')

@dataclass
class BenchConfig:
    method: str = 'perlin'
    t_warmup: int = 1
    t_sample: int = 3
    precision: torch.dtype = torch.float32
    bsize: int = 1
    seq_len: int = 4096
    k: int = 64
    nbf: float = 1

def bench(name, fn, config: BenchConfig):
    sample_count = 0
    try:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        print(f'[{name}] warmup... ', end = '', flush=True)
        t = time.time()
        while True:
            with torch.no_grad(), torch.autocast('cuda', config.precision):
                fn()
            if time.time() - t > config.t_warmup:
                break
        torch.cuda.synchronize()
        print('benchmarking', end = '', flush=True)
        t = time.time()
        last_report = time.time()
        while True:
            with torch.no_grad(), torch.autocast('cuda', config.precision):
                fn()
            sample_count += 1
            if time.time() - t > config.t_sample:
                break
            if time.time() - last_report > 0.5:
                last_report = time.time()
                print('.', end='', flush=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() - start_mem
        elapsed = time.time() - t
    except torch.cuda.OutOfMemoryError as ex: # type: ignore
        mem = 0
        elapsed = 0
    interval = elapsed/(sample_count + 1e-8)
    print(f' done. sampled {sample_count}its. {interval*1000:.2f}ms/it {mem // 1024 // 1024} MB', flush=True)
    return interval, mem

class IndentityXY(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return x

def exam(bench_config: BenchConfig, return_queue: mp.Queue):
    seed()
    
    SEQ_LEN = bench_config.seq_len
    BSIZE = bench_config.bsize
    BENCH_PRECISION = bench_config.precision
    method = bench_config.method
    
    device = torch.device('cuda')

    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = SEQ_LEN

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=bench_config.nbf if method == 'perlin' else 1,
        lora_enabed=False,
        lora_in_approx_enabled=False,
        partial_attention_scaler=True,
        k_flatten=True,
        k=bench_config.k,
    ))
    perlin = BertModel(config).eval()
    for module in perlin.modules():
        if isinstance(module, BertSelfAttention):
            module.perlin_token_merging = False
            module.perlin_token_merging_preserve_ratio = 0.2
            module.perlin_token_merging_ratio = 0.5
            module.perlin_token_merging_score_source = 'probs'
            module.attention_method = method
        if hasattr(module, 'benchmarking'):
            module.benchmarking = True

    attention_mask = torch.ones((BSIZE, SEQ_LEN), dtype=BENCH_PRECISION).to(device)
    # for i in range(attention_mask.shape[0]):
    #     attention_mask[i, random.randint(128, attention_mask.shape[1]-1):] = 0
    hidden_states = torch.randn((BSIZE, SEQ_LEN, config.hidden_size), device=device, dtype=BENCH_PRECISION)
    attention_mask_expand = attention_mask.view(BSIZE, 1, 1, -1).contiguous()
    attention_mask_expand = (1-attention_mask_expand)*(-32000)

    layer = perlin.encoder.layer[0] # type: BertLayer
    attention = layer.attention.self # type: BertSelfAttention
    # attention.teacher_attention_prob = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
    # attention.teacher_attention_score = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
    # attention.teacher_context_layer = torch.rand((BSIZE, SEQ_LEN, config.hidden_size), device=device)
    layer.intermediate = nn.Identity()
    layer.output = IndentityXY()
    layer.attention.output = IndentityXY()
    
    def test_layer():
        layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
    
    layer.to(device)
    result = bench(f'{method},{bench_config.seq_len}{f",{bench_config.k}" if method == "perlin" else ""}', test_layer, bench_config)
    if return_queue is not None:
        return_queue.put(result)

def exam_config(config: BenchConfig):
    q = mp.Queue()
    proc = mp.Process(target=exam, args=(config, q), daemon=True)
    proc.start()
    proc.join()
    return q.get()

def main_methods():
    for method in ['perlin', 'none', 'performer', 'reformer', 'scatterbrain', 'sinkhorn', 'synthesizer']:
        exam_config(BenchConfig(
            method=method
        ))
    
def main_plot():
    precision = torch.float32
    
    baseline_methods = ['none', 'performer', 'reformer', 'sinkhorn', 'synthesizer', 'scatterbrain']
    ts = [2**x for x in range(8, 14)]
    ks = [2**x for x in range(3, 9)]
    # ts = [2**x for x in range(13, 13)]
    # ks = [2**x for x in range(5, 7)]
    
    result_perlin = [
        [
            exam_config(BenchConfig(
                precision=precision,
                method='perlin',
                seq_len=t,
                k=k
            ))
            for t in ts
        ]
        for k in ks
    ]
    
    result_baseline = [
        [
            exam_config(BenchConfig(
                precision=precision,
                method=method,
                seq_len=t,
            ))
            for t in ts
        ]
        for method in baseline_methods
    ]
    
    latencies_baseline = [
        [(x * 1000 if y > 0 else float("nan")) for x, y in result]
        for result in result_baseline
    ]
    latencies_perlin = [
        [(x * 1000 if y > 0 else float("nan")) for x, y in result]
        for result in result_perlin
    ]
    vram_baseline = [
        [(y / (1024**2) if y > 0 else float("nan")) for x, y in result]
        for result in result_baseline
    ]
    vram_perlin = [
         [(y / (1024**2) if y > 0 else float("nan")) for x, y in result]
        for result in result_perlin
    ]
    
    root = './plots/main/benchmark_bert'
    os.makedirs(root, exist_ok=True)
    
    with open(os.path.join(root, 'data.json'), 'w') as f:
        json.dump({
            'latencies_baseline': latencies_baseline,
            'latencies_perlin': latencies_perlin,
            'vram_baseline': vram_baseline,
            'vram_perlin': vram_perlin,
            'ts': ts,
            'ks': ks,
        }, f)
    
    def plot(metric_name, baselines, perlins, ts, ks):
        plt.clf()
        
        for iy, ys in enumerate(baselines):
            plt.plot(ts, ys, label=baseline_methods[iy], linestyle='--', linewidth=0.75)
        for ik, k in enumerate(ks):
            plt.plot(ts, perlins[ik], label=f'k={k}', linewidth=0.75)
        
        plt.title(f'{metric_name}')
        plt.xlabel(f'tokens')
        plt.ylabel(f'{metric_name}')
        plt.yscale('log', base=2)
        plt.xscale('log', base=2)
        plt.grid()
        plt.legend()
        
        path = os.path.join(root, f'{metric_name}.png')
        plt.savefig(path, dpi=300)
        print('saved', path)
    
    plot('latency', latencies_baseline, latencies_perlin, ts, ks)
    plot('vram', vram_baseline, vram_perlin, ts, ks)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    seed()
    main_plot()
    # main_methods()