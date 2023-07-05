import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def collate_fn(items):
    pass

class LRAListOps(Dataset):
    def __init__(self, csv_path, tokenizer: AutoTokenizer) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.load(tokenizer)
        
    def load(self, tokenizer: AutoTokenizer):
        data = []
        
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            spl = line.split('  ')
            source = spl[0]
            target = spl[1]
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

def get_loader(tokenizer: AutoTokenizer, split: str = 'train', batch_size: int = 2):
    csv_path = {
        'train': './src/dataset/_lra_benchmarks/data/output_dir/basic_train.csv',
        'test': './src/dataset/_lra_benchmarks/data/output_dir/basic_test.csv',
        'val': './src/dataset/_lra_benchmarks/data/output_dir/basic_val.csv',
    }[split]
    dataset = LRAListOps(csv_path, tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return loader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    loader = get_loader(tokenizer, split='train', batch_size=2)
    for batch in loader:
        print([(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()])
        break