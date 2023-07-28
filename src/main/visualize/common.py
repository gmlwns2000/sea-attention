import torch
import cv2
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from ...utils import batch_to
from ...models import perlin_bert
from ...trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from matplotlib import cm
import tqdm
import torch.nn.functional as F

ZOOM = 1

def gather_fixed_batch(dataloader: DataLoader, batch_size: int):
    items = [
        dataloader.dataset.__getitem__(i * (len(dataloader.dataset) // batch_size))
        for i in range(batch_size)
    ]
    max_len = max([it['input_ids'].shape[0] for it in items])
    for it in items:
        it['input_ids'] = F.pad(it['input_ids'], (0, max_len-len(it['input_ids'])))
        if 'token_type_ids' in it:
            it['token_type_ids'] = F.pad(it['token_type_ids'], (0, max_len-len(it['token_type_ids'])))
        if 'attention_mask' in it:
            it['attention_mask'] = F.pad(it['attention_mask'], (0, max_len-len(it['attention_mask'])))
    # print([[(k, v.shape) for k, v in it.items()] for it in items])
    return dataloader.collate_fn(items)

def convert_to_colormap(arr: np.ndarray):
    T, T = arr.shape
    arr_min, arr_max = np.min(arr), np.max(arr)
    normalized = (arr - arr_min) / (arr_max - arr_min + 1e-12)
    colormapped = cm.gist_earth(normalized)
    gamma = 0.2
    colormapped = (colormapped / np.max(colormapped)) ** gamma
    im = Image.fromarray((colormapped*255).astype(np.uint8))
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
    
    top = 12*ZOOM
    stacks = np.concatenate([np.zeros((top, stacks.shape[1], stacks.shape[2]), dtype=np.uint8), stacks], axis=0)
    cv2.putText(stacks, f"Layer {idx}", (2*ZOOM, 9*ZOOM), fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=max(1, ZOOM//2), fontScale=max(1, ZOOM//2), color=(0, 255, 0))
    
    return stacks

def process_batch_index(attentions: List[torch.Tensor], i: int, T: int):
    imgs = []
    for ilayer, attn in enumerate(tqdm.tqdm(attentions, dynamic_ncols=True, desc='render.layer')):
        img = process_layer(
            teacher=attn['teacher_attn'][i][:, :T, :T],
            est=attn['estimated_attn'][i][:, :T, :T],
            dense=attn['dense_attn'][i][:, :T, :T],
            partial=attn['partial_attn'][i][:, :T, :T],
            idx=ilayer,
        )
        imgs.append(img)
    return np.concatenate(imgs, axis=0)