from dataset.test_batch_func import make_test_batch

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--subset')
    parser.add_argument('--test-batch-size')
    parser.add_argument('--base-model-type')

    args = parser.parse_args()

    DATASET = args.dataset
    SUBSET = args.subset
    TEST_BATCH_SIZE = args.test_batch_size
    BASE_MODEL_TYPE = args.base_model_type

    make_test_batch(BASE_MODEL_TYPE, DATASET, SUBSET, TEST_BATCH_SIZE)


