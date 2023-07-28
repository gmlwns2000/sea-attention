from torch.utils.data import Dataset, DataLoader
import math
import torch
import os
from datasets import load_dataset

class Wikitext2Dataset(Dataset):
    def __init__(self, subset, tokenizer, stride=2048, max_length=None):
        super().__init__()
        
        if subset == 'valid':
            subset = 'validation'
        
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=subset)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
        self.seq_len = self.encodings.input_ids.size(1)
        self.stride = stride
        self.max_length = max_length
    
    def __len__(self):
        return math.ceil(self.seq_len / self.stride)
    
    def __getitem__(self, idx):
        max_length = self.max_length
        assert max_length > 0
        
        begin_loc = idx * self.stride
        end_loc = min(begin_loc + max_length, self.seq_len)
        trg_len = end_loc - min((idx - 1) * self.stride + max_length, self.seq_len)
        input_ids = self.encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        return {
            'input_ids': input_ids[0],
            'labels': target_ids[0],
            'trg_len': torch.tensor(trg_len),
        }

def get_dataloader(subset, tokenizer, batch_size=1, max_length=None):
    assert max_length is not None
    ds = Wikitext2Dataset(subset, tokenizer, max_length=max_length)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=0, 
        shuffle=subset=='train'
    )

if __name__ == '__main__':
    import transformers
    t = transformers.AutoTokenizer.from_pretrained('facebook/opt-125m')
    loader = get_dataloader('valid', t, batch_size=1, max_length=2048)
    
    for batch in loader:
        print([(k, v.shape) for k, v in batch.items()])