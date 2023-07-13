import torch
import cv2
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from ..utils import batch_to
from ..models import perlin_bert
from ..trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from matplotlib import cm
import torch.nn.functional as F

ZOOM = 8

def gather_fixed_batch(dataloader: DataLoader, batch_size: int):
    items = [
        dataloader.dataset.__getitem__(i * (len(dataloader.dataset) // batch_size))
        for i in range(batch_size)
    ]
    max_len = max([it['input_ids'].shape[0] for it in items])
    for it in items:
        it['input_ids'] = F.pad(it['input_ids'], (0, max_len-len(it['input_ids'])))
        it['attention_mask'] = F.pad(it['attention_mask'], (0, max_len-len(it['attention_mask'])))
        it['token_type_ids'] = F.pad(it['token_type_ids'], (0, max_len-len(it['token_type_ids'])))
    # print([[(k, v.shape) for k, v in it.items()] for it in items])
    return dataloader.collate_fn(items)

def convert_to_colormap(arr: np.ndarray):
    T, T = arr.shape
    im = Image.fromarray((cm.gist_earth((arr-np.min(arr)) / (np.max(arr) - np.min(arr)))*255).astype(np.uint8))
    arr = np.asarray(im)[:, :, :3]
    arr = cv2.resize(arr, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    border = np.ones((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]), dtype=np.uint8)
    border = border * 255
    border[1:-1, 1:-1, :] = arr
    return border

def process_layer(teacher: torch.Tensor, est: torch.Tensor, dense: torch.Tensor, partial: torch.Tensor, idx: int):
    H, T, T = teacher.shape
    
    stacks = []
    for i in range(H):
        stacks.append(np.concatenate([
            convert_to_colormap(teacher[i].cpu().numpy()),
            convert_to_colormap(est[i].cpu().numpy()),
            convert_to_colormap(dense[i].cpu().numpy()),
            convert_to_colormap(partial[i].cpu().numpy()),
        ], axis=0))
    stacks = np.concatenate(stacks, axis=1)
    
    top = 96
    stacks = np.concatenate([np.zeros((top, stacks.shape[1], stacks.shape[2]), dtype=np.uint8), stacks], axis=0)
    cv2.putText(stacks, f"Layer {idx}", (16, 72), fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=4, fontScale=4.0, color=(0, 255, 0))
    
    return stacks

def process_batch_index(attentions: List[torch.Tensor], i: int, T: int):
    imgs = []
    for ilayer, attn in enumerate(attentions):
        img = process_layer(
            teacher=attn['teacher_attn'][i][:, :T, :T],
            est=attn['estimated_attn'][i][:, :T, :T],
            dense=attn['dense_attn'][i][:, :T, :T],
            partial=attn['partial_attn'][i][:, :T, :T],
            idx=ilayer,
        )
        imgs.append(img)
    return np.concatenate(imgs, axis=0)

def main(
    subset = 'mnli',
    checkpoint_path = None,
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
    
    os.makedirs(f"./plots/visualize_glue/{trainer.exp_name}/", exist_ok=True)
    for i in range(len(batch['input_ids'])):
        token_length = int(batch['attention_mask'][i].sum().item())
        img = process_batch_index(attentions, i, token_length)
        path = f"./plots/visualize_glue/{trainer.exp_name}/{i}.png"
        cv2.imwrite(path, img)
        print('processed', path)
    
    print('accuracy', trainer.evaluate())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subset', type=str, default='mnli')
    parser.add_argument('--checkpoint', type=str, default=None)
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'subset': args.subset,
        'checkpoint_path': args.checkpoint,
    })
    
    main(**kwargs)