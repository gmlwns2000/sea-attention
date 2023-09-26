import copy
from dataclasses import dataclass
import math
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
from ..models.perlin_opt import OPTModel, OPTAttention, OPTDecoderLayer
from ..utils import seed, get_bench
from torch import nn
import json

plt.style.use('seaborn-bright')
plt.rcParams['font.family'] = 'Noto Sans'

@dataclass
class BenchConfig:
    method: str = 'perlin'
    t_warmup: int = 1
    t_sample: int = 3
    precision: torch.dtype = torch.float32
    bsize: int = 1
    seq_len: int = 4096
    k: int = 64
    w: int = None
    nbf: float = 8
    trace: bool = True
    causal: bool = False
    n_hash: int = 8
    opt_model: str = 'facebook/opt-125m'

def bench(name, fn, config: BenchConfig, on_warmup=None):
    sample_count = 0
    try:
        torch.cuda.synchronize()
        print(f'[{name}] warmup... ', end = '', flush=True)
        t = time.time()
        while True:
            with torch.no_grad(), torch.autocast('cuda', config.precision):
                fn()
            if time.time() - t > config.t_warmup:
                break
        if on_warmup is not None:
            on_warmup()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        print('benchmarking', end = '', flush=True)
        elapsed = 0
        last_report = time.time()
        t = time.time()
        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad(), torch.autocast('cuda', config.precision):
                fn()
            end.record()
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end) / 1000
            
            sample_count += 1
            if time.time() - t > config.t_sample:
                break
            if time.time() - last_report > 0.5:
                last_report = time.time()
                print('.', end='', flush=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() - start_mem
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

    if bench_config.w is None:
        pred_len = 128
        # pred_len = 64
        # if bench_config.seq_len >= 2048:
        #     pred_len = 128
        # elif bench_config.seq_len >= 4096:
        #     pred_len = 256
        # elif bench_config.seq_len >= 8192:
        #     pred_len = 256
        # elif bench_config.seq_len >= 16384:
        #     pred_len = 256
    else:
        pred_len = bench_config.w

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=bench_config.nbf if method == 'perlin' or (method == 'performer' and bench_config.causal) else 1,
        lora_enabed=False,
        lora_in_approx_enabled=False,
        partial_attention_scaler=True,
        k_flatten=True,
        k=bench_config.k,
        attention_predictor_length=pred_len
    ))
    if not bench_config.causal:
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.max_position_embeddings = SEQ_LEN
        perlin = BertModel(config).eval()
    else:
        config = AutoConfig.from_pretrained(bench_config.opt_model)
        # config.max_position_embeddings = SEQ_LEN
        perlin = OPTModel(config).eval()
    for module in perlin.modules():
        if isinstance(module, BertSelfAttention):
            module.perlin_token_merging = False
            module.perlin_token_merging_preserve_ratio = 0.2
            module.perlin_token_merging_ratio = 0.5
            module.perlin_token_merging_score_source = 'probs'
            module.attention_method = method
        if isinstance(module, OPTAttention):
            module.attention_method = method
            module.perlin_reformer_atten.n_hashes = bench_config.n_hash
        if hasattr(module, 'benchmarking'):
            module.benchmarking = True

    if not bench_config.causal:
        attention_mask = torch.ones((BSIZE, SEQ_LEN), dtype=BENCH_PRECISION).to(device)
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
    else:
        attention_mask = (torch.arange(0, SEQ_LEN).view(SEQ_LEN, 1) >= torch.arange(0, SEQ_LEN).view(1, SEQ_LEN)) * 1.0
        hidden_states = torch.randn((BSIZE, SEQ_LEN, config.hidden_size), device=device, dtype=BENCH_PRECISION)
        attention_mask_expand = attention_mask.to(device).view(1, 1, SEQ_LEN, SEQ_LEN)
        
        layer = perlin.decoder.layers[0] # type: OPTDecoderLayer
        fc1 = nn.Identity()
        fc1.weight = layer.fc1.weight
        layer.fc1 = fc1
        layer.fc2 = nn.Identity()
        out_proj = nn.Identity()
        out_proj.weight = layer.self_attn.out_proj.weight
        layer.self_attn.out_proj = out_proj
    
    def test_layer():
        layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
    
    layer.to(device)
    
    get_bench().disabled = False
    get_bench().synchronize = True
    get_bench().reset_temp_buffers()
    get_bench().reset_trace()
    get_bench().reset_measures()
    
    def on_warmup():
        get_bench().reset_temp_buffers()
        get_bench().reset_trace()
        get_bench().reset_measures()
    
    if method == 'perlin' and bench_config.trace:
        bench(f'{method},{bench_config.seq_len}{f",{bench_config.k}" if method == "perlin" else ""} (trace)', test_layer, bench_config, on_warmup=on_warmup)
        msg = get_bench().format_tracetree()
        if len(msg) > 0: print(msg)
    
    get_bench().disabled = True
    get_bench().synchronize = False
    
    result_interval, result_mem = bench(f'{method},{bench_config.seq_len}{f",{bench_config.k}" if method == "perlin" else ""}{f",{bench_config.nbf}" if method == "perlin" else ""}{f",{bench_config.w}" if method == "perlin" else ""}', test_layer, bench_config)
    # print(result_interval, BSIZE)
    result_interval = result_interval / BSIZE
    result_mem = result_mem / BSIZE
    
    if return_queue is not None:
        return_queue.put((result_interval, result_mem))

def exam_config(config: BenchConfig):
    q = mp.Queue()
    proc = mp.Process(target=exam, args=(config, q), daemon=True)
    proc.start()
    proc.join()
    return q.get()

# BASELINES = ['none', 'performer', 'reformer', 'scatterbrain', 'sinkhorn', 'synthesizer']
BASELINES = ['none', 'cosformer', 'performer', 'reformer', 'scatterbrain', 'sinkhorn', 'synthesizer']
# BASELINES = ['none',]

def main_methods():
    for method in BASELINES:
        exam_config(BenchConfig(
            method=method
        ))
    
def measure_and_dump():
    TRACE = True
    precision = torch.float32
    
    baseline_methods = BASELINES
    ts = [2**x for x in range(10, 16)]
    # ks = [2**x for x in range(3, 8)]
    ks = [32, 64, 128,]
    # ks = [32]
    # ts = [1024*32]
    # ks = [32]
    # ts = [2**x for x in range(13, 13)]
    # ks = [2**x for x in range(5, 7)]
    
    result_perlin = [
        [
            exam_config(BenchConfig(
                precision=precision,
                method='perlin',
                seq_len=t,
                k=k,
                trace=TRACE,
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
                trace=False,
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
            'baseline_methods': baseline_methods,
            'ts': ts,
            'ks': ks,
        }, f)

def load_and_plot():
    baseline_methods = BASELINES
    root = './plots/main/benchmark_bert'
    os.makedirs(root, exist_ok=True)
    
    with open(os.path.join(root, 'data.json'), 'r') as f:
        data = json.load(f)
    latencies_baseline = data['latencies_baseline']
    latencies_perlin = data['latencies_perlin']
    vram_baseline = data['vram_baseline']
    vram_perlin = data['vram_perlin']
    ts = data['ts']
    ks = data['ks']
    
    def plot(filename, title, ylabel, baselines, perlins, ts, ks):
        plt.figure(figsize=(4.5,3.5))
        
        NAMES = {
            'none': 'None',
            'cosformer': 'Cosformer',
            'performer': 'Performer',
            'reformer': 'Reformer',
            'scatterbrain': 'ScatterBrain',
            'sinkhorn': 'Sinkhorn',
            'synthesizer': 'Synthesizer',
        }
        
        LINESTYLE = {
            'none': '--',
            'cosformer': ':',
            'performer': ':',
            'reformer': ':',
            'scatterbrain': ':',
            'sinkhorn': ':',
            'synthesizer': ':',
            'perlin': '-',
        }
        
        MARKERS = {
            'none': '>',
            'cosformer': '+',
            'performer': 'v',
            'reformer': '^',
            'scatterbrain': 'x',
            'sinkhorn': 'h',
            'synthesizer': 'd',
        }
        
        MARKER_SIZE = {
            'none': 5,
            'cosformer': 5,
            'performer': 5,
            'reformer': 5,
            'scatterbrain': 5,
            'sinkhorn': 5,
            'synthesizer': 3,
            'perlin': 7,
        }
        
        COLORS = {
            'none': 'mediumslateblue',
            'cosformer': 'gray',
            'performer': 'lightcoral',
            'reformer': 'greenyellow',
            'scatterbrain': 'turquoise',
            'sinkhorn': 'lightskyblue',
            'synthesizer': 'gold',
            'perlin_0': 'red',
            'perlin_1': 'darkorange',
            'perlin_2': 'limegreen',
        }
        
        xs_est = ts
        ys_est = copy.deepcopy(baselines[BASELINES.index('none')])
        y_slope = ys_est[1] / ys_est[0]
        y_last = ys_est[1]
        assert not math.isnan(y_last)
        for i in range(2, len(ys_est)):
            if math.isnan(ys_est[i]):
                ys_est[i] = ys_est[i-1] * y_slope
            y_slope = ys_est[i] / y_last
            y_last = ys_est[i]
        
        plt.plot(
            xs_est, 
            ys_est, 
            linestyle=':', 
            linewidth=0.5,
            label='None (Trend)',
            marker='4',
            markersize = MARKER_SIZE['none'],
            color=COLORS['none'],
        )
        
        for iy, ys in enumerate(baselines):
            plt.plot(
                ts, 
                ys, 
                label=NAMES[baseline_methods[iy]], 
                linestyle=LINESTYLE[baseline_methods[iy]], 
                linewidth=0.75,
                marker=MARKERS[baseline_methods[iy]],
                markersize=MARKER_SIZE[baseline_methods[iy]],
                color=COLORS[baseline_methods[iy]],
            )
        for ik, k in enumerate(ks):
            plt.plot(
                ts, 
                perlins[ik], 
                label=f'Ours (k={k})', 
                linewidth=1.0,
                marker='*',
                markersize=MARKER_SIZE['perlin'],
                color=COLORS[f'perlin_{ik}'],
            )
        
        plt.title(f'{title}', fontweight=500)
        plt.xlabel(f'Seq. Len.', fontweight=500)
        plt.ylabel(f'{ylabel}', fontweight=500)
        plt.yscale('log', base=2)
        plt.xscale('log', base=2)
        plt.grid()
        plt.legend(fontsize=6, ncols=2)
        
        path = os.path.join(root, f'{filename}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print('saved', path)
        path = os.path.join(root, f'{filename}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print('saved', path)
    
    plot(
        'exp_latency', 
        'Latency', 'ms/it', 
        latencies_baseline, 
        latencies_perlin, 
        ts, 
        ks
    )
    plot(
        'exp_vram', 
        'Peak VRAM Usage', 
        'MB', 
        vram_baseline, 
        vram_perlin, 
        ts, 
        ks
    )

def main_plot():
    measure_and_dump()
    load_and_plot()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    seed()
    main_plot()
    # main_methods()