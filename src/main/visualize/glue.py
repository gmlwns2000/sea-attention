import torch
import cv2
import os
import argparse

from ...utils import batch_to
from ...models import perlin_bert
from ...trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options

from .common import (
    gather_fixed_batch,
    process_batch_index,
)

bool2int = lambda x: 1 if x else 0
LAYER_ID = 0
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
    
    if 'benchmark_for_sparse' in kwargs:
        for module in bert.modules():
            if hasattr(module, 'benchmarking'):
                module.benchmarking = kwargs['benchmark_for_sparse']
            if hasattr(module, 'layer_id'):
                global LAYER_ID
                module.layer_id = LAYER_ID
                LAYER_ID += 1


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
            dense_attn = module.last_perlin_dense_probs
            partial_attn = module.last_perlin_partial_probs
            attentions.append({
                'teacher_attn': teacher_attn,
                'estimated_attn': estimated_attn,
                'dense_attn': dense_attn,
                'partial_attn': partial_attn,
            })            
        
    r = bool2int(kwargs['perlin_colsel'])
    m = kwargs['perlin_colsel_method']
    p = bool2int(kwargs['perlin_colsel_mask_in_probs'])
    root = f"./plots/visualize_glue/colsel{r}/"
    if kwargs['perlin_colsel']:
        root += f'/{m}_mprobs{p}'
    os.makedirs(root, exist_ok=True)
    for i in range(len(batch['input_ids'])):
        token_length = int(batch['attention_mask'][i].sum().item())
        # token_length = batch['input_ids'].shape[-1]
        img = process_batch_index(attentions, i, token_length)
        
        path = root + f"/{i}.png"
        cv2.imwrite(path, img)
        print('processed', path)
    
    if evaluate:
        print('accuracy', trainer.evaluate())

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