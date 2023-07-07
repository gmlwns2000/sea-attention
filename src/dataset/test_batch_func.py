import os
import torch
from ..trainer import bert_glue_trainer as bert_glue_trainer

bert_glue_task_to_valid = {
    "cola": "validation",
    "mnli": "validation_matched",
    "mrpc": "test",
    "qnli": "validation",
    "qqp": "validation",
    "rte": "validation",
    "sst2": "validation",
    "stsb": "validation",
    "wnli": "validation",
    "bert": "validation",
}

def make_test_batch(base_model_type, dataset, subset, test_batch_size):
    # tokenizer, batch_size, tast_to_valid
    if base_model_type == 'bert':
        if dataset == "glue":
            _, tokenizer = bert_glue_trainer.get_base_model(subset)
            valid_loader = bert_glue_trainer.get_dataloader(
            subset, 
            tokenizer, 
            test_batch_size, 
            split=bert_glue_task_to_valid[subset])
            # TODO check
            test_batch = valid_loader.collate_fn(
                valid_loader.dataset.__getitem__(i)
             for i in len(test_batch))
            return test_batch

    # elif base_model_type =="glue":
    # ...

    save_test_batch(dataset, subset, test_batch, test_batch_size)


def save_test_batch(dataset, subset, test_batch, test_batch_size):
    os.makedirs(f'./saves/dataset/{dataset}/{subset}/', exist_ok=True)
    path = f'./saves/dataset/{dataset}/{subset}/test_batch_size_{test_batch_size}.pth'
    print(f'test_batch saved in "{path}"')
    torch.save({
        'test_batch': test_batch
    }, path)

def load_test_batch(dataset, subset, test_batch_size):
    state = torch.load(f'./saves/dataset/{dataset}/{subset}/test_batch_size_{test_batch_size}.pth', map_location='cpu')
    test_batch = state['test_batch']
    del state
    return test_batch