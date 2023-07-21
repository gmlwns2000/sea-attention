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

T_WARMUP = 2
T_SAMPLE = 5
BENCH_PRECISION = torch.float16
BENCH_PRECISION = torch.float32
BSIZE = 1
SEQ_LEN = 4096

def bench(name, fn):
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
            with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                fn()
            if time.time() - t > T_WARMUP:
                break
        torch.cuda.synchronize()
        print('benchmarking', end = '', flush=True)
        t = time.time()
        last_report = time.time()
        while True:
            with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                fn()
            sample_count += 1
            if time.time() - t > T_SAMPLE:
                break
            if time.time() - last_report > 0.5:
                last_report = time.time()
                print('.', end='', flush=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() - start_mem
    except torch.cuda.OutOfMemoryError as ex: # type: ignore
        mem = 0
    elapsed = time.time() - t
    interval = elapsed/(sample_count + 1e-8)
    print(f' done. sampled {sample_count}its. {interval*1000:.2f}ms/it {mem // 1024 // 1024} MB', flush=True)
    return interval, mem

def exam(method):
    device = torch.device('cuda')

    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = SEQ_LEN

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=8 if method == 'perlin' else 1,
        lora_enabed=False,
        lora_in_approx_enabled=False,
        partial_attention_scaler=True,
        k_flatten=True,
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

    attention_mask = torch.ones((BSIZE, SEQ_LEN)).to(device)
    for i in range(attention_mask.shape[0]):
        attention_mask[i, random.randint(128, attention_mask.shape[1]-1):] = 0
    hidden_states = torch.randn((BSIZE, SEQ_LEN, config.hidden_size), device=device, dtype=BENCH_PRECISION)
    attention_mask_expand = attention_mask.view(BSIZE, 1, 1, -1).contiguous()
    attention_mask_expand = (1-attention_mask_expand)*(-32000)

    layer = perlin.encoder.layer[0] # type: BertLayer
    attention = layer.attention.self # type: BertSelfAttention
    attention.teacher_attention_prob = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
    attention.teacher_attention_score = torch.rand((BSIZE, 12, SEQ_LEN, SEQ_LEN), device=device)
    attention.teacher_context_layer = torch.rand((BSIZE, SEQ_LEN, config.hidden_size), device=device)
    
    def test_layer():
        layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
    
    layer.to(device)
    return bench(method, test_layer)

import torch.multiprocessing as mp

def main():
    mp.set_start_method('spawn')
    
    print(f"config(fp={BENCH_PRECISION}, bsize={BSIZE}, seq_len={SEQ_LEN})")
    
    for method in ['perlin', 'none', 'performer', 'reformer', 'scatterbrain', 'sinkhorn', 'synthesizer']:
        proc = mp.Process(target=exam, args=(method,), daemon=True)
        proc.start()
        proc.join()

if __name__ == '__main__':
    seed()
    main()