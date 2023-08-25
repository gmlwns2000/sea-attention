from torch.utils.data import Dataset, DataLoader
import math
import torch
import os
from datasets import load_dataset
import tqdm
from torch.utils.data.distributed import DistributedSampler

class Wikitext2Dataset(Dataset):
    def __init__(self, subset, tokenizer, stride=2048, max_length=None, strided_indexing=None):
        super().__init__()
        
        if subset == 'valid':
            subset = 'validation'
        if subset in ['validation', 'test'] and strided_indexing is None:
            strided_indexing = True
        self.strided_indexing = strided_indexing
        
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=subset)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
        self.seq_len = self.encodings.input_ids.size(1)
        self.stride = stride
        self.max_length = max_length
        self.check_last_shape = subset == 'train'
        self.last_shape = None
    
    def __len__(self):
        if self.strided_indexing:
            return math.ceil(self.seq_len / self.stride)
        else:
            # return self.seq_len - self.stride * 2
            return self.seq_len - self.stride
    
    def __getitem__(self, idx):
        max_length = self.max_length
        assert max_length > 0
        
        if not self.strided_indexing:
            # idx = idx + self.stride
            begin_loc = idx
        else:
            begin_loc = idx * self.stride
        
        end_loc = min(begin_loc + max_length, self.seq_len)
        trg_len = end_loc - min(begin_loc - self.stride + max_length, self.seq_len)
        
        input_ids = self.encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        if self.check_last_shape:
            if self.last_shape is not None:
                assert self.last_shape == input_ids.shape
            self.last_shape = input_ids.shape
        
        return {
            'input_ids': input_ids[0],
            'labels': target_ids[0],
            'trg_len': torch.tensor(trg_len),
        }

def get_dataloader(subset, tokenizer, batch_size=1, max_length=None, local_rank=0, world_size=1):
    assert max_length is not None
    ds = Wikitext2Dataset(subset, tokenizer, stride=max_length, max_length=max_length)
    use_shuffle = subset=='train'
    
    if world_size > 1:
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            num_workers=0, 
            sampler=DistributedSampler(
                dataset=ds,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=use_shuffle,
            )
        )
    else:
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            num_workers=0, 
            shuffle=use_shuffle
        )

if __name__ == '__main__':
    import transformers
    t = transformers.AutoTokenizer.from_pretrained('facebook/opt-125m')
    # loader = get_dataloader('train', t, batch_size=1, max_length=768)
    loader = get_dataloader('valid', t, batch_size=1, max_length=768)
    
    for batch in tqdm.tqdm(loader):
        # print([(k, v.shape) for k, v in batch.items()])
        pass