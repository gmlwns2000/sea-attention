import os
from dataset.test_batch_generator import load_test_batch
from ..models import hf_bert as berts
from ..models import perlin_bert as perlin
from transformers import BertConfig
import torch
from ..trainer.bert_glue_trainer import get_dataloader, get_base_model
from ..utils import batch_to, seed
import warnings

bool2int = lambda x: 1 if x else 0

def load_model(): # NOTE : optimizer not included
    state = torch.load(PATH, map_location='cpu')

    epoch = state['epoch']
    step = state['step']
    lr = state['lr']
    
    model = get_model_init(BASE_MODEL_TYPE, TASK_TYPE)
    base_model = get_basem_init(BASE_MODEL_TYPE, TASK_TYPE)

    model.load_state_dict(state['model'], strict=False)
    base_model.load_state_dict(state['base_model'], strict=True)
    model.eval()
    base_model.eval()

    del state
    return epoch, step, lr, model, base_model

def get_attns_img(DATASET, SUBSET, model, base_model, img_title):
    dense_attns_img = None
    sparse_attns_img = None

    if ATTENTION_METHOD=="base":
        dense_attn_probs = sample_attentions_basem(DATASET, SUBSET, base_model, dense_attn = True)
        dense_attns_img = attentions_to_img(dense_attn_probs, img_title)
        assert dense_attns_img is not None
        return dense_attns_img, None
    elif ATTENTION_METHOD=='perlin':
        dense_attn_probs = sample_attentions_model(DATASET, SUBSET, model, base_model, dense_attn = True)
        sparse_attn_probs = sample_attentions_model(DATASET, SUBSET, model, base_model, dense_attn = False)
        dense_attns_img = attentions_to_img(dense_attn_probs, img_title)
        sparse_attns_img = attentions_to_img(sparse_attn_probs, img_title)
        assert dense_attns_img is not None and sparse_attns_img is not None
        return dense_attns_img, sparse_attns_img
    elif ATTENTION_METHOD == 'performer':
        sparse_attn_probs = sample_attentions_model(DATASET, SUBSET, model, base_model, dense_attn = False)
        sparse_attns_img = attentions_to_img(sparse_attn_probs, img_title)
        assert sparse_attns_img is not None
        return None, sparse_attns_img
    else:
        raise Exception("check ATTNETION_METHOD")

def save_eval(obj, folder_path, file_path): # TODO
    os.makedirs(folder_path, exist_ok=True)
    path = folder_path + file_path
    print(f'Evaluation: save {path}')
    torch.save(obj, path)
    print(f'Evaluation: saved {path}')

def main():
    epoch, step, lr, model, base_model = load_model()
    
    # NOTE(JIN) add other eval tasks
    if 'viz_attentions' in EVAL_TYPES:
        img_title = f"ep{epoch}_st{step}_lr{lr}"
        dense_attns_img, sparse_attns_img = get_attns_img(DATASET, SUBSET, model, base_model, img_title)

        assert PATH[:8] is './saves/'
        FOLDER_PATH = PATH.replace('./saves/','./plots/')
        folder_idx = FOLDER_PATH.rindex('/')
        FOLDER_PATH = FOLDER_PATH[:folder_idx+1] # ~~/~/~/

        if dense_attns_img is not None:
            save_eval(dense_attns_img, FOLDER_PATH+f"{ATTENTION_METHOD}/", f"dense_attentions_{img_title}.png")
        if sparse_attns_img is not None:
            save_eval(sparse_attns_img, FOLDER_PATH+f"{ATTENTION_METHOD}/", f"sparse_attentions_{img_title}.png")
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='bert', type=str)
    parser.add_argument('--task-type', default='seqclassification', type=str)
    parser.add_argument('--dataset', default='glue', type=str)
    parser.add_argument('--subset', default='mnli', type=str)
    
    parser.add_argument('--method', default='perlin', type=str) # in ["base", "perlin", "performer", longformer, bigbird, sinkhorn, synthesizer, reformer, ...]
    parser.add_argument('--layerwise', action='store_true') # default: False, type --layerwise to make True
    parser.add_argument('--k-relwise', action='store_true')
    parser.add_argument('--redraw-proj', action='store_true')

    parser.add_argument('--path', default='',type=str)

    # evaluation
    parser.add_argument('--eval-types', default='viz_attentions', type=str) # TODO update default : put every tasks in default
    
    args = parser.parse_args()
    
    BASE_MODEL_TYPE = args.base_model
    TASK_TYPE = args.task_type
    DATASET = args.dataset
    SUBSET = args.subset

    ATTENTION_METHOD = args.method
    # dependent on attention_method
    PERLIN_LAYERWISE = args.layerwise
    PERLIN_K_RELWISE = args.k_relwise
    PERLIN_REDRAW_PROJ = args.redraw_proj

    PATH = args.path
    
    # inlcudes all model eval
    EVAL_TYPES = args.eval_types

    main()