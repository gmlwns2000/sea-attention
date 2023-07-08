from ...dataset.test_batch_save_load import save_test_batch
from ...trainer import bert_glue_trainer as bert_glue_trainer
from datasets import load_dataset

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
            loaded_dataset = load_dataset(dataset, subset, split=bert_glue_task_to_valid[subset])
            len_dataset = len(loaded_dataset)
            batch_size = len_dataset//test_batch_size
            # breakpoint()
            valid_loader = bert_glue_trainer.get_dataloader(
            subset, 
            tokenizer, 
            batch_size, # stepping size
            split=bert_glue_task_to_valid[subset],
            making_test_batch = True)
            
            # TODO check - torch has __getitem__?, how does it work?
            test_batch = valid_loader.collate_fn(
                [valid_loader.dataset.__getitem__(i*batch_size) # TODO check idx
             for i in range(test_batch_size)])
            
            assert len(test_batch['labels']) == test_batch_size

    # elif base_model_type =="glue":
    # ...
    
    save_test_batch(dataset, subset, FOR_EVAL, test_batch, test_batch_size)
    return test_batch

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="glue", type=str)
    parser.add_argument('--subset', default="mnli", type=str)
    parser.add_argument('--for-eval', action='store_true') # default=false -> default train
    parser.add_argument('--test-batch-size', default=1, type=int)
    parser.add_argument('--base-model-type', default='bert', type=str)

    args = parser.parse_args()

    DATASET = args.dataset
    SUBSET = args.subset
    FOR_EVAL = args.for_eval
    TEST_BATCH_SIZE = args.test_batch_size
    BASE_MODEL_TYPE = args.base_model_type

    if not FOR_EVAL:
        assert TEST_BATCH_SIZE == 1

    make_test_batch(BASE_MODEL_TYPE, DATASET, SUBSET, TEST_BATCH_SIZE)


