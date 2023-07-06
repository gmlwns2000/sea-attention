from ...models import perlin_bert

def get_model_self_attn(base_model_type):
    model_self_attn = {
        "bert" : perlin_bert.BertSelfAttention,
        # "gpt" : perlin_gpt.GPT2Attention,
    }[base_model_type]
    return model_self_attn