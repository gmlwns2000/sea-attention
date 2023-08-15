# causal checking

# pass attention
import argparse
import warnings

from ....trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options

from ....models.perlin_opt.perlin_opt import OPTDecoder
from ....models.perlin_attention import PerlinAttention
from ....models.hf_bert import BertConfig
from ....models.perlin_attention.config import PerlinAttentionConfig, register_default_config


import torch

# path ='./debug/opt/attn_params.pth'

# state = torch.load(path, map_location='cpu')
# q = state['q']
# k = state['k']
# v = state['v']
# q_for_atten = state['q_for_atten']
# k_for_atten = state['k_for_atten']
# v_for_atten = state['v_for_atten']
# q_for_score = state['q_for_score']
# k_for_score = state['k_for_score']
# attention_mask = state['attention_mask']
# attention_scores_truth = state['attention_scores_truth']
# context_layer_truth = state['context_layer_truth']
# last_state = state['last_state']

def main(
    checkpoint_path = None,
    evaluate = False,
    use_cache = False,
    **kwargs
):
    #### changeable conditions
    N=1
    H=12
    T=2048
    HID=64

    #### params : TODO random is okay?
    # tokenizer -> input_ids -> attention input들 구해야
    
    query_layer = torch.randn(N, H, T, HID)
    key_layer = torch.randn(N, H, T, HID)
    value_layer = torch.randn(N, H, T, HID)
    query_layer_for_atten = query_layer.clone()
    key_layer_for_atten = key_layer.clone()
    value_layer_for_atten = value_layer.clone()
    query_layer_for_score = query_layer.clone()
    key_layer_for_score = value_layer.clone()

    mask = torch.ones(N, T) # modify
    warnings.warn(f"mask({mask.shape}) {mask}")
    input_shape = (N, T)
    inputs_embeds = torch.randn(N, T, T)
    past_key_values_length = 0
    causal_attention_mask = OPTDecoder._prepare_decoder_attention_mask(-1, mask, input_shape, inputs_embeds, past_key_values_length)

    attention_scores_truth = torch.randn(N, H, T, T)
    context_layer_truth = torch.randn(N, T, H*HID)
    last_state = None

    #### config
    perlin_config = PerlinAttentionConfig(
        k=kwargs["perlin_k"],
        causal=True,
        use_cache=use_cache
        )
    print(perlin_config)
    warnings.warn(f"perlin_k {perlin_config.k}")
    warnings.warn(f"k_flatten {perlin_config.k_flatten}")
    warnings.warn(f"k_flatten_dim {perlin_config.k_flatten_dim}")
    warnings.warn(f"causal {perlin_config.causal}")
    warnings.warn(f"use_cache {perlin_config.use_cache}")

    #### attention
    attention = PerlinAttention(
        config=BertConfig(num_attention_heads=H), # WHY 
        perlin_config=perlin_config
    )

    output = attention(
        q=query_layer,
        k=key_layer,
        v=value_layer,
        q_for_atten=query_layer_for_atten,
        k_for_atten=key_layer_for_atten,
        v_for_atten=value_layer_for_atten,
        q_for_score=query_layer_for_score,
        k_for_score=key_layer_for_score,
        attention_mask=causal_attention_mask,
        attention_scores_truth=attention_scores_truth,
        context_layer_truth=context_layer_truth,
        last_state=last_state,
        )

    print(output.partial_attention_probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use-cache', action='store_true')

    add_perlin_model_options(parser)

    args = parser.parse_args()
    args.k_flatten_dim = 'causal_batch'

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate,
        'use_cache': args.use_cache
    })
    
    if kwargs["perlin_k"]==7:
        print("INIT perlin_k was : ", kwargs['perlin_k'])
        kwargs['perlin_k'] = 64
        print("PERLIN_K is changed to : ", kwargs['perlin_k'])
    main(**kwargs)