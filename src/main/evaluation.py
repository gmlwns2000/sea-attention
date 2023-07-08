import os
from ..main.visualization.attentions_to_img import attentions_to_img
import torch
from ..models.initialize_model import get_model_init
from ..models.sampling.sampling_attentions import sample_attentions_basem, sample_attentions_model

bool2int = lambda x: 1 if x else 0

def load_model(
        path, 
        base_model_type, 
        task_type, 
        dataset, 
        subset): # NOTE : optimizer not included
    state = torch.load(path, map_location='cpu')

    epoch = state['epoch']
    step = state['step']
    lr = state['lr']
    
    model, base_model = get_model_init(base_model_type, task_type, dataset, subset)

    model.load_state_dict(state['model'], strict=False)
    base_model.load_state_dict(state['base_model'], strict=True)
    model.eval()
    base_model.eval()

    del state
    return epoch, step, lr, model, base_model

def get_attns_img(
        base_model_type, 
        dataset, 
        subset, 
        attention_method, 
        model, 
        base_model, 
        img_title,
        test_batch_size,
        for_eval):
    model.eval()
    base_model.eval()
    dense_attns_img = None
    sparse_attns_img = None
    
    if attention_method=="base":
        dense_attn_probs = sample_attentions_basem(
            for_eval,
            test_batch_size,
            dataset, 
            subset, 
            base_model)
        dense_attns_img = attentions_to_img(
            dense_attn_probs, 
            dataset, 
            subset, 
            img_title, 
            test_batch_size, 
            for_eval)
        assert dense_attns_img is not None
        return dense_attns_img, None
    elif attention_method=='perlin':
        dense_attn_probs = sample_attentions_model(
            for_eval,
            test_batch_size,
            base_model_type, 
            dataset, 
            subset, 
            model, 
            base_model, 
            viz_dense_attn = True)
        sparse_attn_probs = sample_attentions_model(
            for_eval,
            test_batch_size,
            base_model_type, 
            dataset, 
            subset, 
            model, 
            base_model, 
            viz_dense_attn = False)
        dense_attns_img = attentions_to_img(
            dense_attn_probs, 
            dataset, 
            subset, 
            img_title, 
            test_batch_size,
            for_eval)
        sparse_attns_img = attentions_to_img(
            sparse_attn_probs, 
            dataset, 
            subset, 
            img_title, 
            test_batch_size,
            for_eval)
        assert dense_attns_img is not None and sparse_attns_img is not None
        return dense_attns_img, sparse_attns_img
    elif attention_method == 'performer':
        sparse_attn_probs = sample_attentions_model(
            for_eval,
            test_batch_size,
            base_model_type, 
            dataset, 
            subset, 
            model, 
            base_model,
            viz_dense_attn = False)
        sparse_attns_img = attentions_to_img(
            sparse_attn_probs, 
            dataset, 
            subset, 
            img_title, 
            test_batch_size,
            for_eval)
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

def save_fig(plot, folder_path, file_path): # TODO
    os.makedirs(folder_path, exist_ok=True)
    path = folder_path + file_path
    print(f'Evaluation: save fig {path}')
    plot.savefig(path, dpi=96)
    print(f'Evaluation: saved fig {path}')

def main():
    epoch, step, lr, model, base_model = load_model(
        PATH, 
        BASE_MODEL_TYPE, 
        TASK_TYPE, 
        DATASET, 
        SUBSET)
    
    # NOTE(JIN) add other eval tasks
    if 'viz_attentions' in EVAL_TYPES:
        img_title = f"ep{epoch}_st{step}_lr{lr}"
        dense_attns_img, sparse_attns_img = get_attns_img(
            BASE_MODEL_TYPE, 
            DATASET, 
            SUBSET, 
            ATTENTION_METHOD, 
            model, 
            base_model, 
            img_title,
            TEST_BATCH_SIZE,
            FOR_EVAL)

        assert PATH[:8] == './saves/'
        FOLDER_PATH = PATH.replace('./saves/','./plots/')
        folder_idx = FOLDER_PATH.rindex('/')
        FOLDER_PATH = FOLDER_PATH[:folder_idx+1] # ~~/~/~/

        if dense_attns_img is not None:
            save_fig(dense_attns_img, FOLDER_PATH+f"{ATTENTION_METHOD}/", f"dense_attns_{img_title}.png")
        if sparse_attns_img is not None:
            save_fig(sparse_attns_img, FOLDER_PATH+f"{ATTENTION_METHOD}/", f"sparse_attns_{img_title}.png")
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='bert', type=str)
    parser.add_argument('--task-type', default='seqclassification', type=str)
    parser.add_argument('--dataset', default='glue', type=str)
    parser.add_argument('--subset', default='mnli', type=str)
    parser.add_argument('--test-batch-size', default=10, type=int)

    
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
    TEST_BATCH_SIZE = args.test_batch_size
    assert TEST_BATCH_SIZE % 2==0

    ATTENTION_METHOD = args.method
    # dependent on attention_method
    PERLIN_LAYERWISE = args.layerwise
    PERLIN_K_RELWISE = args.k_relwise
    PERLIN_REDRAW_PROJ = args.redraw_proj

    PATH = args.path
    
    # inlcudes all model eval
    EVAL_TYPES = args.eval_types

    FOR_EVAL = True
    assert FOR_EVAL

    main()