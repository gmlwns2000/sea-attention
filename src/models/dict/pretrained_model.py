def pretrained_model_dict(base_model_type, dataset, subset):
    if base_model_type == 'bert':
        if dataset == "glue":
            pretrained_model = {
            "cola": "textattack/bert-base-uncased-CoLA",
            "mnli": "yoshitomo-matsubara/bert-base-uncased-mnli",
            "mrpc": "textattack/bert-base-uncased-MRPC",
            # "mrpc": "M-FAC/bert-tiny-finetuned-mrpc",
            "qnli": "textattack/bert-base-uncased-QNLI",
            "qqp": "textattack/bert-base-uncased-QQP",
            "rte": "textattack/bert-base-uncased-RTE",
            "sst2": "textattack/bert-base-uncased-SST-2",
            "stsb": "textattack/bert-base-uncased-STS-B",
            "wnli": "textattack/bert-base-uncased-WNLI",
            "bert": "bert-base-uncased",
            }[subset]
            return pretrained_model
        # elif ... other dataset: 
    # elif base_model_type == "gpt":
    #     ...