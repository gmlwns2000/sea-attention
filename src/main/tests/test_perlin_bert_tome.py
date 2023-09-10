import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random
from transformers import AutoConfig
from ...models.hf_bert import BertModel as TeacherBertModel
from ...models import perlin_bert
from ...models.perlin_bert import BertModel, BertSelfAttention

device = 0

perlin_bert.PERLIN_PERFORMER_NB_FACTOR = 16
config = AutoConfig.from_pretrained('bert-base-uncased')
config.max_position_embeddings = 2048
teacher = TeacherBertModel(config).to(device).eval()
bert = BertModel(config).to(device).eval()
for module in bert.modules():
    if isinstance(module, BertSelfAttention):
        module.perlin_token_merging = True
        module.perlin_token_merging_preserve_ratio = 0.2
        module.perlin_token_merging_ratio = 0.5
        module.perlin_token_merging_score_source = 'key'
        module.attention_method = 'performer'
        module.benchmarking = True

input_ids = torch.randint(0, 10000, (2, 2048)).to(device)
attention_mask = torch.ones((2, 2048)).to(device)
for i in range(attention_mask.shape[0]):
    attention_mask[i, random.randint(5, attention_mask.shape[1]-1):] = 0

N_SAMPLE = 100

with torch.no_grad(), torch.autocast('cuda', torch.float16):
    output_teacher = teacher(input_ids=input_ids, attention_mask=attention_mask)
    output = bert(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    torch.cuda.synchronize()

t = time.time()
for i in tqdm.tqdm(range(N_SAMPLE)):
    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        output = bert(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
torch.cuda.synchronize()
t_bert = time.time() - t

t = time.time()
for i in tqdm.tqdm(range(N_SAMPLE)):
    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        output_teacher = teacher(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
t_teacher = time.time() - t

print(
    output.last_hidden_state.shape,
    output_teacher.last_hidden_state.shape,
    t_teacher, 
    t_bert, 
    t_bert / t_teacher,
    torch.cuda.max_memory_allocated() // 1024 // 1024,
    torch.nn.functional.mse_loss(output.last_hidden_state, output_teacher.last_hidden_state)
)