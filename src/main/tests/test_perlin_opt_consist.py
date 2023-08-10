import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random, json
from transformers import AutoConfig
from ...models.hf_bert import BertLayer
from ...models.hf_bert import BertModel as TeacherBertModel
from ...models import perlin_opt
from ...models.perlin_attention import PerlinSelfAttention, PerlinAttentionOutput
from ...models.perlin_opt import OPTForCausalLM, OPTDecoderLayer, OPTAttention
from ...models.perlin_attention.config import PerlinAttentionConfig, register_default_config
from ...utils import seed, get_bench, strify
import torch.nn.functional as F
from .common_opt import init

seed()
get_bench().disabled = False

# device = torch.device('cuda')

# config = AutoConfig.from_pretrained('facebook/opt-125m')

# model = OPTForCausalLM(config).to(device).eval()
# for module in model.modules():
#     if isinstance(module, OPTAttention):
#         module.attention_method = 'perlin'
trainer, model, tokenizer = init(skip_init_loaders=True)
device = trainer.device
config = model.config

def set_benchmark(model, v):
    for module in model.modules():
        if hasattr(module, 'benchmarking'):
            module.benchmarking = v

N = 1
H = model.config.num_attention_heads
HID = 64
SEQ_LEN = 2048
BENCH_PRECISION = torch.float32
FP_MIN = torch.finfo(torch.float32).min

q = torch.randn((N, H, SEQ_LEN, HID), device=device, dtype=BENCH_PRECISION)
k = q.clone()
v = q.clone()
attention_mask = (torch.arange(SEQ_LEN).view(1, SEQ_LEN) > torch.arange(SEQ_LEN).view(SEQ_LEN, 1)) * FP_MIN
attention_mask = attention_mask.to(device).view(1, 1, SEQ_LEN, SEQ_LEN)
attention_scores_truth = torch.randn((1, H, SEQ_LEN, SEQ_LEN), device=device)
context_layer_truth = torch.randn((1, SEQ_LEN, H*HID), device=device)

layer = model.model.decoder.layers[0] # type: OPTDecoderLayer
attention = layer.self_attn.perlin_self_attention

def samples(
    sample_from_benchmark=False, 
    score_mask_keys=[],
    masked_score_keys=[],
):
    get_bench().activate_temp_buffers = sample_from_benchmark
    with torch.no_grad():
        output = attention(
            query=None,
            key=None,
            value=None,
            hidden_states=context_layer_truth,
            query_layer=q,
            key_layer=k,
            value_layer=v,
            attention_mask=attention_mask,
            attention_scores_truth=attention_scores_truth,
            context_layer_truth=context_layer_truth,
        ) # type: PerlinAttentionOutput
    get_bench().activate_temp_buffers = False
    
    def postproc(key, sample):
        if key in score_mask_keys:
            sample = (sample > -1).float()
        if key in masked_score_keys:
            sample = sample * (sample > -32000)
        return sample
    
    samples = {
        # 'output.partial_attention_mask': output.partial_attention_mask,
        # 'output.partial_attention_probs': output.partial_attention_probs,
        'output.estimated_attention_probs_m': output.estimated_attention_probs_m,
        'output.context_layer': output.context_layer,
    }
    for key in samples:
        samples[key] = postproc(key, samples[key])
    keys = []
    
    buffers = get_bench().buffers
    for key in buffers:
        for i, sample in enumerate(buffers[key]):
            sample = postproc(key, sample)
            sample_key = f'temp.{key}[{i}]'
            keys.append(sample_key)
            samples[sample_key] = sample
    keys += [
        # 'output.partial_attention_mask',
        # 'output.partial_attention_probs',
        'output.estimated_attention_probs_m',
        'output.context_layer',
    ]
    
    get_bench().reset_temp_buffers()
    
    return keys, samples

set_benchmark(model, False)
sample_b0_keys, sample_b0 = samples(
    sample_from_benchmark=True,
    score_mask_keys=[
        'output.partial_attention_mask',
        'partial_attention_mask_before_interp',
        'partial_attention_mask',
    ],
    masked_score_keys=[
        'partial_attention_scores',
    ]
)
set_benchmark(model, True)
sample_b1_keys, sample_b1 = samples(
    sample_from_benchmark=True,
)

keys = sample_b1_keys
just_width = max([len(k) for k in keys])

def exam(name, a, b, thresh=1e-5):
    name = name.ljust(just_width)
    
    print(f'\033[94m> exam {name}\033[0m')
    if a is None or b is None:
        print(f'  \033[90m[skip]\033[0m {name} : is None {strify(a)} {strify(a)}. skip checking')
        return
    
    if a.is_sparse:
        a = a.to_dense()
    if b.is_sparse:
        b = b.to_dense()
    if a.is_sparse_csr:
        raise Exception(name)
    if b.is_sparse_csr:
        raise Exception(name)

    if a.shape != b.shape:
        print(f'  \033[93m[warn]\033[0m {name} : shape is mismatch {a.shape} != {b.shape}')
        a = a.view(b.shape)
    
    loss = F.mse_loss(a, b, reduction='sum')
    
    if loss > thresh:
        os.makedirs('./saves/tests/test_perlin_opt_consist/', exist_ok=True)
        path = f'./saves/tests/test_perlin_opt_consist/{name.strip()}.pth'
        torch.save({
            'a':a, 'b':b, 'name': name
        }, path)
        print(f'  \033[91m[fail]\033[0m {name} : is not matched. loss {loss}. dumped {path}.')
    else:
        print(f'  \033[92m[pass]\033[0m {name} : is matched.     loss {loss}.')

for key in keys:
    exam(key, sample_b0.get(key, None), sample_b1.get(key, None))

#############################

N_WARNUP = 30
N_SAMPLE = 500
get_bench().synchronize = True
get_bench().activate_temp_buffers = False

get_bench().reset_temp_buffers()
get_bench().reset_trace()
get_bench().reset_measures()
set_benchmark(model, True)
for _ in tqdm.tqdm(range(N_WARNUP)):
    samples()
for _ in tqdm.tqdm(range(N_SAMPLE)):
    samples()

data = get_bench().todict()
# print(data)

root = get_bench().traced_callstack
total_time = data[root.name]
def format_tree_percent(self, indent=0):
    spaces = "--" * indent
    messages =  [f"{spaces}> {self.name} ({data[self.name]*1000:.2f} ms, {data[self.name] / total_time * 100:.2f}%)"]
    for child in self.children:
        messages.append(format_tree_percent(child, indent+1))
    return "\n".join(messages)
tree = format_tree_percent(root)
print(tree)