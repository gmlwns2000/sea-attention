from .list_ops import get_loader as get_list_ops_loader

def get_loader(subset: str, split: str, batch_size: int):
    if subset == 'listops':
        return get_list_ops_loader(split=split, batch_size=batch_size)
    else:
        raise Exception()

def get_loaders(subset: str, batch_size: int):
    train = get_loader(subset, 'train', batch_size)
    test = get_loader(subset, {
        'listops': 'val',
    }[subset], batch_size)
    return train, test

if __name__ == '__main__':
    pass