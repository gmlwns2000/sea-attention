import pandas as pd
import torch, os
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import tqdm

def make_word_tokenizer(allowed_words, lowercase_input=False, allow_unk=True):
    # make distinct
    allowed_words = sorted(list(set(allowed_words)))
    allowed_words = dict({c: (i+2) for i, c in enumerate(allowed_words)})
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def tokenizer(x_str, max_length):
        # note: x_str is not batched
        if lowercase_input:
            x_str = x_str.lower()

        x = x_str.split()
        x = x[:max_length]
        n = len(x)
        # mask = ([1] * n) + ([0] * (max_length - n))
        mask = torch.empty((max_length,), dtype=torch.long)
        mask[:n] = 1
        mask[n:] = 0
        ids = list(map(lambda c: allowed_words[c] if c in allowed_words else UNK_TOKEN, x))
        input_ids = torch.empty((max_length), dtype=torch.long)
        input_ids[:n] = torch.tensor(ids)
        input_ids[n:] = PAD_TOKEN
        if not allow_unk:
            assert UNK_TOKEN not in input_ids, "unknown words are not allowed by this tokenizer"
        return {
            'input_ids': input_ids, 
            'attention_mask': mask
        }

    tokenizer.vocab_size = len(allowed_words) + 2
    return tokenizer

def get_tokenizer():
    return make_word_tokenizer(list('0123456789') + ['[', ']', '(', ')', 'MIN', 'MAX', 'MEDIAN', 'SUM_MOD'])

class LRAListOps(Dataset):
    def __init__(self, csv_path, tokenizer: AutoTokenizer, max_length=2048) -> None:
        super().__init__()

        self.max_length = max_length
        self.csv_path = csv_path
        self.load(tokenizer)
        
    def load(self, tokenizer: AutoTokenizer):
        os.makedirs('./cache/dataset/lra_benchmark/', exist_ok=True)
        csv_name = os.path.basename(self.csv_path)
        cache_path = f'./cache/dataset/lra_benchmark/{csv_name}.pth'
        if os.path.exists(cache_path):
            print('cache find from', cache_path, end='...')
            self.data = torch.load(cache_path)
            print('loaded!')
        else:
            self.data = []
            
            data = pd.read_csv(self.csv_path, delimiter='\t')
            for i in tqdm.tqdm(range(len(data)), desc='LRAListOps.load', dynamic_ncols=True):
                if i > 500: break
                row = data.iloc[i]
                source = row.Source
                inputs = tokenizer(
                    source, 
                    max_length=self.max_length, 
                    # return_tensors='pt', 
                    # truncation=True, 
                    # padding='max_length'
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                target = row.Target
                self.data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(target, dtype=torch.long)
                })

            torch.save(self.data, cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_loader(split: str = 'train', batch_size: int = 2):
    tokenizer = get_tokenizer()
    
    csv_path = {
        'train': './src/dataset/lra_benchmarks/lra_pytorch/datasets/lra_release/listops-1000/basic_train.tsv',
        'test': './src/dataset/lra_benchmarks/lra_pytorch/datasets/lra_release/listops-1000/basic_test.tsv',
        'val': './src/dataset/lra_benchmarks/lra_pytorch/datasets/lra_release/listops-1000/basic_val.tsv',
    }[split]
    dataset = LRAListOps(csv_path, tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # collate_fn=collate_fn,
        shuffle=split=='train'
    )
    return loader

if __name__ == '__main__':
    loader = get_loader(split='train', batch_size=2)
    for batch in loader:
        print(*[(k, v[0]) for k, v in batch.items()], sep='\n')
        print([(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()])
        break
    