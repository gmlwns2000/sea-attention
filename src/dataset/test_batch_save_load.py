import os
import torch

bool2int = lambda x: 1 if x else 0

def save_test_batch(dataset, subset, for_eval, test_batch, test_batch_size):
    if not for_eval:
        eval_or_train = "train"
    else:
        eval_or_train = "eval"
    os.makedirs(f'./saves/dataset/test_batch/{dataset}/{subset}/{eval_or_train}/', exist_ok=True)
    path = f'./saves/dataset/test_batch/{dataset}/{subset}/{eval_or_train}/batch_size_{test_batch_size}.pth'
    print(f'test_batch saved in "{path}"')
    torch.save({
        'test_batch': test_batch
    }, path)

def load_test_batch(dataset, subset, for_eval, test_batch_size):
    if not for_eval:
        eval_or_train = "train"
    else:
        eval_or_train = "eval"
    path = f'./saves/dataset/test_batch/{dataset}/{subset}/{eval_or_train}/batch_size_{test_batch_size}.pth'
    state = torch.load(path, map_location='cpu')
    print(f'test_batch load "{path}"')
    test_batch = state['test_batch']
    del state
    return test_batch