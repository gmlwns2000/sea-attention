import os
from typing import List
from matplotlib import cm

import numpy as np
from main.test_batch_generator import load_test_batch, make_save_test_batch

from trainer.perlin_trainer import GlueTrainer, add_perlin_model_options, parse_perlin_model_options
from .visualization.attentions_to_img import attentions_to_img
import torch
import cv2
from PIL import Image

from ..models.initialize_model import create_model
from ..models.sampling.sampling_attentions import sample_attentions_basem, sample_attentions_model

from ..models.perlin_bert import perlin_bert
from torch.utils.data import DataLoader, Dataset

ZOOM = 8

def get_test_batch(dataloader: DataLoader, subset: str, test_batch_size: int):
    # NOTE path in test_batch_save_load
    if not os.path.exists(os.path.join(os.getcwd(), './saves/dataset/test_batch/', f'glue_{subset}_bs{test_batch_size}.pth')):
        make_save_test_batch(dataloader, 'glue', subset, test_batch_size)
    return load_test_batch('glue',subset, test_batch_size)

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
    test_batch_size = 10,
    **kwargs
):
    trainer = GlueTrainer(
        subset=subset,
        **kwargs
    )
    trainer.load(checkpoint_path)

    test_batch = get_test_batch(trainer.valid_loader, subset, test_batch_size)

    bert = trainer.model
    teacher = trainer.base_model

    img_title = ""
    if hasattr(trainer, 'epoch'):
        epoch = trainer.epoch
        img_title+=f"ep{epoch}/"
    if hasattr(trainer, 'step'):
        step = trainer.step
        img_title+=f"st{step}"
    bert.eval()
    teacher.eval()

    with torch.no_grad():
        teacher(**test_batch)
        bert['teacher'] = teacher
        bert(**test_batch)
    
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
    
    os.makedirs(f"./plots/visualize_bert_glue/", exist_ok=True)
    for i in range(len(test_batch['input_ids'])):
        token_len = int(test_batch['attention_mask'][i].sum().item())
        img = process_batch_index(attentions, i, token_len)
        path = f"./plots/visualize_bert_glue/{i}.png"
        cv2.imwrite(path, img)
        print('img saved in ', path)

    print('accuracy', trainer.evaluate())
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', default='mnli', type=str)
    parser.add_argument('--test-batch-size', default=10, type=int)
    parser.add_argument('--checkpoint', default=None,type=str)
    add_perlin_model_options(parser)

    args = parser.parse_args()
    print(args)

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'subset' : args.subset,
        'test_batch_size' : args.test_batch_size,
        'checkpoint_path' : args.checkpoint
    })

    main(**kwargs)