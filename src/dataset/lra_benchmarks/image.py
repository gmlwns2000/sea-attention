import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os, tqdm
from glob import glob
from itertools import cycle
import numpy as np
from functools import reduce

def get_tokenizer():
    PAD_TOKEN = 0
    def pixel_tokenizer(x, max_length):
        x = x.flatten()
        x = x[:max_length]
        n = len(x)
        
        ids = torch.tensor(x) + 1
        input_ids = torch.empty((max_length), dtype=torch.long)
        input_ids[:n] = ids
        input_ids[n:] = PAD_TOKEN
        
        mask = torch.empty((max_length,), dtype=torch.long)
        mask[:n] = 1
        mask[n:] = 0
        
        return {
            'input_ids': input_ids, 
            'attention_mask': mask
        }
    
    pixel_tokenizer.vocab_size = 256 + 1
    
    return pixel_tokenizer

class LRAImage(Dataset):
    def __init__(self, tokenizer, path, max_length = 1024):
        self.path = path
        self.max_length = max_length
        self.load(tokenizer)
    
    def load(self, tokenizer):
        os.makedirs('./cache/dataset/lra_benchmark/', exist_ok=True)
        pathname = os.path.basename(sorted(self.path)[0])
        cache_path = f'./cache/dataset/lra_benchmark/image_{pathname}.pth'
        if os.path.exists(cache_path):
            print('cache find from', cache_path, end='...')
            self.data = torch.load(cache_path)
            print('loaded!')
        else:
            self.load_from_path(tokenizer)
            torch.save(self.data, cache_path)
    
    def load_from_path(self, tokenizer):
        self.data = []
        
        print("loading cifar-10 data...")
        data_dicts = [self.unpickle(path) for path in self.path]
        print("assembling cifar-10 files..")
        data = reduce((lambda x, y: {
            b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
            b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)
        }), data_dicts)
        data[b'data'] = data[b'data'].reshape((-1, 3, 1024))
        
        for i in tqdm.tqdm(range(len(data[b'data'])), desc='LRAImage.load', dynamic_ncols=True):
            r, g, b = data[b'data'][i]
            source = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
            inputs = tokenizer(source, max_length=self.max_length)
            target = data[b'labels'][i]
            inputs['labels'] = torch.tensor(target, dtype=torch.long)
            self.data.append(inputs)
    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_loader(split: str = 'train', batch_size: int = 2):
    tokenizer = get_tokenizer()
    
    path = {
        'train': [f"./src/dataset/lra_benchmarks/lra_pytorch/datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)],
        'test': ["./src/dataset/lra_benchmarks/lra_pytorch/datasets/cifar-10-batches-py/test_batch"]
    }[split]
    dataset = LRAImage(tokenizer, path)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # collate_fn=collate_fn,
        shuffle=split=='train'
    )
    return loader

if __name__ == '__main__':
    loader = get_loader(split='train', batch_size=2)
    loader = get_loader(split='test', batch_size=2)
    for batch in loader:
        print(*[(k, v[0]) for k, v in batch.items()], sep='\n')
        print([(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()])
        break