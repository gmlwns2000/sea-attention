import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random, json
from transformers import AutoConfig
from ...models.hf_bert import BertLayer
from ...models.hf_bert import BertModel as TeacherBertModel
from ...models import perlin_bert
from ...models.perlin_bert import BertModel, BertSelfAttention
from ...models.perlin_attention.config import PerlinAttentionConfig, register_default_config
from ...utils import seed, get_bench

def main():
    get_bench().synchronize = True
    N_WARMUP = 30
    N_SAMPLE = 100
    BENCH_PRECISION = torch.float16
    BSIZE = 1
    SEQ_LEN = 4096
    layerwise = True
    BENCHMARK = True
    
    device = torch.device('cuda')

    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = 4096
    teacher = TeacherBertModel(config).to(device).eval()

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=8,
        lora_enabed=False,
        lora_in_approx_enabled=False,
        partial_attention_scaler=False,
        k_flatten=True,
    ))
    perlin = BertModel(config).to(device).eval()
    for module in perlin.modules():
        if isinstance(module, BertSelfAttention):
            module.perlin_token_merging = False
            module.perlin_token_merging_preserve_ratio = 0.2
            module.perlin_token_merging_ratio = 0.5
            module.perlin_token_merging_score_source = 'probs'
            module.attention_method = 'perlin'
        if hasattr(module, 'benchmarking'):
            module.benchmarking = BENCHMARK

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=1,
    ))
    performer = BertModel(config).to(device).eval()
    for module in performer.modules():
        if isinstance(module, BertSelfAttention):
            module.attention_method = 'performer'
        if hasattr(module, 'benchmarking'):
            module.benchmarking = BENCHMARK


    input_ids = torch.randint(0, 10000, (BSIZE, SEQ_LEN)).to(device)
    attention_mask = torch.ones((BSIZE, SEQ_LEN)).to(device)
    for i in range(attention_mask.shape[0]):
        attention_mask[i, random.randint(128, attention_mask.shape[1]-1):] = 0

    if not layerwise:
        with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
            output_teacher = teacher(input_ids=input_ids, attention_mask=attention_mask)
            output = perlin(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
            output_perf = performer(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    torch.cuda.synchronize()

    def bench(fn):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        for i in tqdm.tqdm(range(N_WARMUP), desc='warmup'):
            with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                fn()
        torch.cuda.synchronize()
        t = time.time()
        for i in tqdm.tqdm(range(N_SAMPLE), desc='benchmarking'):
            with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                fn()
        torch.cuda.synchronize()
        return time.time() - t, torch.cuda.max_memory_allocated() - start_mem
    
    hidden_states = torch.randn((BSIZE, SEQ_LEN, teacher.config.hidden_size), device=device, dtype=BENCH_PRECISION)
    attention_mask_expand = attention_mask.view(BSIZE, 1, 1, -1).contiguous()
    
    def test_bert():
        if layerwise:
            layer = teacher.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            teacher(input_ids=input_ids, attention_mask=attention_mask)
    
    def test_perlin():
        if layerwise:
            layer = perlin.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            perlin(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    
    def test_performer():
        if layerwise:
            layer = perlin.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            performer(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    
    t_bert, m_bert = bench(lambda: test_bert())
    t_perlin, m_perlin = bench(lambda: test_perlin())
    t_performer, m_performer = bench(lambda: test_performer())

    bench_result = get_bench().todict()
    print(
        # output.last_hidden_state.shape, 
        # output_teacher.last_hidden_state.shape, 
        json.dumps({k: (v/bench_result['perlin'])*100 for k, v in bench_result.items()}, indent=2),
        f'timer_bert: {t_bert}s, mem_bert: {m_bert // 1024 // 1024}MB', 
        f'time peformer: {t_performer}s, mem_performer: {m_performer // 1024 // 1024}MB', 
        f'time_perlin: {t_perlin}s, mem_perlin: {m_perlin // 1024 // 1024}MB', 
        f'speedup w.r.t performer: {t_performer / t_perlin}',
        f'speedup w.r.t bert: {t_bert / t_perlin}',
        f'max_mem: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB',
        # f'error: {torch.nn.functional.mse_loss(output.last_hidden_state, output_teacher.last_hidden_state)}',
        sep='\n'
    )
    
    return t_bert, t_performer, t_perlin

if __name__ == '__main__':
    seed()
    main()