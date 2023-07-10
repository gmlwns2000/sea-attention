import torch
from torch.utils.data import Dataset, DataLoader
import os, tqdm
from glob import glob
from itertools import cycle
import numpy as np

def make_char_tokenizer(allowed_chars, lowercase_input=False):
    allowed_chars = sorted(list(set(allowed_chars)))
    allowed_chars = dict({c: (i+1) for i, c in enumerate(allowed_chars)})
    # print(allowed_chars)
    PAD_TOKEN = 0

    def tokenizer(x, max_length):
        x = x[:max_length]
        if lowercase_input:
            x = x.lower()
        n = len(x)
        
        mask = torch.empty((max_length,), dtype=torch.long)
        mask[:n] = 1
        mask[n:] = 0
        ids = list(map(lambda c: allowed_chars[c], x))
        input_ids = torch.empty((max_length), dtype=torch.long)
        input_ids[:n] = torch.tensor(ids)
        input_ids[n:] = PAD_TOKEN
        return {
            'input_ids': input_ids, 
            'attention_mask': mask
        }

    tokenizer.vocab_size = len(allowed_chars) + 1
    return tokenizer

def get_tokenizer():
    return make_char_tokenizer(''.join(chr(i) for i in range(256)))

class LRAText(Dataset):
    def __init__(self, tokenizer, path, max_length = 1024):
        self.path = path
        self.max_length = max_length
        self.load(tokenizer)
    
    def load(self, tokenizer):
        os.makedirs('./cache/dataset/lra_benchmark/', exist_ok=True)
        pathname = os.path.basename(self.path)
        cache_path = f'./cache/dataset/lra_benchmark/text_{pathname}.pth'
        if os.path.exists(cache_path):
            print('cache find from', cache_path, end='...')
            self.data = torch.load(cache_path)
            print('loaded!')
        else:
            self.load_from_path(tokenizer)
            torch.save(self.data, cache_path)
    
    def load_from_path(self, tokenizer):
        self.data = []
        
        neg_path = os.path.join(self.path, "neg")
        pos_path = os.path.join(self.path, "pos")
        neg_inputs = zip(glob(os.path.join(neg_path, "*.txt")), cycle([0]))
        pos_inputs = zip(glob(os.path.join(pos_path, "*.txt")), cycle([1]))
        files = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        for i in tqdm.tqdm(range(len(files)), desc='LRAText.load', dynamic_ncols=True):
            filename = files[i]
            with open(filename[0], 'r', encoding='ascii', errors='ignore') as fo:
                source = fo.read()
            inputs = tokenizer(source, max_length=self.max_length)
            target = int(filename[1])
            inputs['labels'] = torch.tensor(target, dtype=torch.long)
            
            self.data.append(inputs)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def get_loader(split: str = 'train', batch_size: int = 2):
    tokenizer = get_tokenizer()
    
    path = {
        'train': './src/dataset/lra_benchmarks/lra_pytorch/datasets/aclImdb/train', 
        'test': './src/dataset/lra_benchmarks/lra_pytorch/datasets/aclImdb/test'
    }[split]
    dataset = LRAText(tokenizer, path)
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
    