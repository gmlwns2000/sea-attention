from .list_ops import get_loader as get_list_ops_loader
from .text import get_loader as get_text_loader
from .image import get_loader as get_image_loader

def get_loader(subset: str, split: str, batch_size: int):
    if subset == 'listops':
        return get_list_ops_loader(split=split, batch_size=batch_size)
    elif subset == 'text':
        return get_text_loader(split=split, batch_size=batch_size)
    elif subset == 'image':
        return get_image_loader(split=split, batch_size=batch_size)
    else:
        raise Exception()

def get_loaders(subset: str, batch_size: int):
    train = get_loader(subset, 'train', batch_size)
    test = get_loader(subset, {
        'listops': 'val',
        'text': 'test',
        'image': 'test',
    }[subset], batch_size)
    return train, test

if __name__ == '__main__':
    pass