from ..models import hf_bert as berts
from ..models import perlin_bert as perlin
from transformers import BertConfig
import torch
from ..trainer.bert_glue_trainer import get_dataloader, get_base_model
from ..utils import batch_to, seed
import wandb
import warnings

bool2int = lambda x: 1 if x else 0

task_to_valid = { # TODO temporarily implemented
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

def get_config(dataset): # TODO temporarily implemented
    checkpoint = {
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
    }[dataset]

    config = BertConfig.from_pretrained(checkpoint)
    return config

class Main():
    def __init__(self) -> None:
        self.perlin_before_topk = PERLIN_BEFORE_TOPK
        self.trainer_name = _TRAINER_NAME
        self.perlin_mode = _MODE if 'perlin' or 'performer' else None
        self.epoch = None
        self.viz_batch = None
        self.loss_details = None
        self.accuracy = None
        # self.config = None
        
        self.device = 0
        
        try: # TODO(JIN): model, base_model initialization are task specific
            print(f"Plot] dataset: {DATASET}, subset: {SUBSET}")
            if _MODE == 'base':
                self.message = f"Plot] Load({_MODE})"
                self.model = berts.BertForSequenceClassification(get_config(SUBSET))
            elif _MODE == 'perlin':
                self.message = f"Plot] Load({_MODE}) lw_{PERLIN_LAYERWISE}, kf_{PERLIN_K_RELWISE}, bftk_{PERLIN_BEFORE_TOPK}"
                self.model = perlin.BertForSequenceClassification(get_config(SUBSET))
            elif _MODE == 'performer':
                self.message = f"Plot] Load({_MODE}) lw_{PERLIN_LAYERWISE}"
                self.model = perlin.BertForSequenceClassification(get_config(SUBSET))
            else:
                raise Exception("Plot] check the --mode.")
        except Exception as ex:
            print('error while load', ex)
        print(self.message)
        self.base_model = berts.BertForSequenceClassification(get_config(SUBSET)) # unlock
        
        self.model.to(self.device)
        self.base_model.to(self.device)
        
        # self.optimizer TODO
    
    from .plot import plot_attentions_all_layer
    
    def load(self):
        ##### TODO temporarily implemented
        self.epoch = -1
        _, self.tokenizer = get_base_model(SUBSET)
        
        self.valid_loader = get_dataloader(SUBSET, self.tokenizer, 16, split=task_to_valid[SUBSET])
        
        for batch in self.valid_loader:
            self.viz_batch = batch_to(batch, self.device)
            break
        #####
        state = torch.load(_PATH, map_location='cpu')
        
        self.model.load_state_dict(state['model'], strict=False) # TODO check strict=False
        self.base_model.load_state_dict(state['base_model'], strict=False) # TODO check strict=False
        # self.optimizer TODO

        # self.epoch = state['epoch'] # TODO unlock comments
        # self.viz_batch = state['viz_batch']
        # self.loss_details = state['loss']
        # self.accuracy = state['accuracy']
        # self.config = state['config'] # TODO delete get_config and use this
        
        self.model.eval()
        self.base_model.eval()
        
        del state
    def main(self):
        project_name = f"[main] {self.message}"
        wandb.init(
             project=project_name
        )
        warnings.warn(project_name+"!")
        
        self.load()
        self.plot_attentions_all_layer(current_state=f"loaded_{self.epoch+1}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='glue', type=str)
    parser.add_argument('--subset', default='mnli', type=str)
    
    parser.add_argument('--mode', default='perlin', type=str) # in ["base", "perlin", "performer"]
    parser.add_argument('--layerwise', action='store_true') # default: False, type --layerwise to make True
    parser.add_argument('--k-relwise', action='store_true')
    parser.add_argument('--before-topk', action='store_true') # for 'perlin'
    
    args = parser.parse_args()
    
    DATASET = args.dataset
    SUBSET = args.subset
    
    _MODE = args.mode
    PERLIN_LAYERWISE = args.layerwise
    PERLIN_K_RELWISE = args.k_relwise
    PERLIN_BEFORE_TOPK = args.before_topk
    if _MODE == 'base':
        _TRAINER_NAME = 'bert_glue_trainer'
    elif _MODE == 'perlin' or _MODE == 'performer':
        _TRAINER_NAME = f'perlin_trainer_kf{bool2int(PERLIN_K_RELWISE)}_lw{bool2int(PERLIN_LAYERWISE)}_{_MODE}' # TODO temporarily implemented
        # _TRAINER_NAME = trainer_name=f'perlin_trainer_lw{bool2int(PERLIN_LAYERWISE)}_rw{bool2int(PERLIN_K_RELWISE)}_bf{bool2int(PERLIN_BEFORE_TOPK)}_{PERLIN_MODE}' # TODO use this
    else:
        raise Exception("Plot: path doesn't exist.")
    _PATH = f'./saves/trainer/hj/{_TRAINER_NAME}/checkpoint_{SUBSET}.pth' # NOTE(JIN) path hj
    
    main_class = Main()
    main_class.main()