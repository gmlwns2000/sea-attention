import os

import torch
from torch.utils.data import DataLoader

def make_save_test_batch(valid_loader: DataLoader, dataset: str, subset: str, test_batch_size: int):
    # tokenizer, batch_size, tast_to_valid
    if dataset == "glue":
        test_batch = valid_loader.collate_fn(
            [valid_loader.dataset.__getitem__(i*(len(valid_loader.dataset)//test_batch_size))
            for i in range(test_batch_size)])
        
        assert len(test_batch['labels']) == test_batch_size
    
    save_test_batch(dataset, subset, test_batch, test_batch_size)
    return test_batch

def save_test_batch(dataset:str, subset:str, test_batch: torch.Tensor, test_batch_size: int):
    os.makedirs(f'./saves/dataset/test_batch/', exist_ok=True)
    path = f'./saves/dataset/test_batch/{dataset}_{subset}_bs{test_batch_size}.pth'
    print(f'test_batch saved in "{path}"')
    torch.save({
        'test_batch': test_batch
    }, path)

def load_test_batch(dataset:str, subset:str, test_batch_size:int):
    path = f'./saves/dataset/test_batch/{dataset}_{subset}_bs{test_batch_size}.pth'
    state = torch.load(path, map_location='cpu')
    print(f'test_batch load "{path}"')
    test_batch = state['test_batch']
    del state
    return test_batch