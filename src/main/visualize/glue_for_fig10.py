import torch
import cv2
import os
import argparse

from ...utils import batch_to
from ...models import perlin_bert
from ...trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options

from .common import (
    gather_fixed_batch,
)

def main(
    subset = 'mnli',
    checkpoint_path = None,
    evaluate = False,
    **kwargs
):
    trainer = GlueTrainer(
        subset=subset,
        **kwargs
    )
    trainer.load(path=checkpoint_path)
    
    batch = gather_fixed_batch(trainer.valid_loader, 10)
    batch = batch_to(batch, trainer.device)
    
    teacher = trainer.base_model
    bert = trainer.model
    
    teacher.eval()
    bert.eval()
    
    with torch.no_grad():
        teacher(**batch)
        batch['teacher'] = teacher
        bert(**batch)
    
    attentions = []
    for module in bert.modules():
        if isinstance(module, perlin_bert.BertSelfAttention):
            teacher_attn = module.teacher_attention_prob
            estimated_attn = module.last_perlin_estimated_probs
            estimated_attn_m = module.perlin_output.estimated_attention_probs_m
            dense_attn = module.last_perlin_dense_probs
            partial_attn = module.last_perlin_partial_probs
            partial_attention_mask=module.perlin_output.partial_attention_mask
            partial_attention_mask_m=module.perlin_output.partial_attention_mask_m
            attentions.append({
                'teacher_attn': teacher_attn.cpu(),
                'estimated_attn': estimated_attn.cpu(),
                'dense_attn': dense_attn.cpu(),
                'partial_attn': partial_attn.cpu(),
                'estimated_attn_m':estimated_attn_m.cpu(),
                'partial_attention_mask':partial_attention_mask.cpu(),
                'partial_attention_mask_m':partial_attention_mask_m.cpu()
            })
    torch.save({
        'teacher_attn': attentions[1]['teacher_attn'],
        'estimated_attn':attentions[1]['estimated_attn'],
        'estimated_attn_m':attentions[1]['estimated_attn_m'],
        'dense_attn': attentions[1]['dense_attn'], 
        'partial_attn':attentions[1]['partial_attn'], 
        'partial_attention_mask':attentions[1]['partial_attention_mask'], 
        'partial_attention_mask_m':attentions[1]['partial_attention_mask_m'],
        'token_length': batch['attention_mask'][7].sum().item()}, 
        './debug/bert_viz.pth') # layer1 'estimated_attn_m':attentions[1]['estimated_attn_m'],

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subset', type=str, default='mnli')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'subset': args.subset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate
    })
    
    main(**kwargs)
