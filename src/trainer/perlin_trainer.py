import warnings
import torch
from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

bool2int = lambda x: 1 if x else 0

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        global PERLIN_LAYERWISE, PERLIN_MODE, PERLIN_K_RELWISE, PERLIN_BEFORE_TOPK

        task_to_batch_size['mnli'] = 16 if not PERLIN_LAYERWISE else 24

        if PERLIN_MODE=="perlin":
            if PERLIN_BEFORE_TOPK == True:
                self.proj_type = "perlin_before"
            elif PERLIN_BEFORE_TOPK == False:
                self.proj_type = "perlin_after"
        elif PERLIN_MODE == "performer":
            self.proj_type = "performer"
        else:
            raise Exception(f"Perlin_trainer] check PERLIN_MODE {PERLIN_MODE}")
        
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=True,
            trainer_name=f'perlin_trainer_lw{bool2int(PERLIN_LAYERWISE)}_rw{bool2int(PERLIN_K_RELWISE)}_bf{bool2int(PERLIN_BEFORE_TOPK)}_{PERLIN_MODE}',
            using_kd=(not PERLIN_LAYERWISE) and (PERLIN_MODE != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps=2000,
            lr = 1e-4,
            epochs = 20,
            proj_type = self.proj_type
        )
        
        self.perlin_mode = PERLIN_MODE
        self.perlin_before_topk = PERLIN_BEFORE_TOPK
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.perlin_mode = PERLIN_MODE
                module.perlin_k_relwise = PERLIN_K_RELWISE
                module.perlin_before_topk = PERLIN_BEFORE_TOPK
        
        if PERLIN_LAYERWISE:
            for module in self.model.modules():
                if isinstance(module, perlin.BertSelfAttention):
                    module.perlin_layerwise = True
                    module.perlin_lora_enabled = True
            
            for name, param in self.model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='perlin', type=str)
    parser.add_argument('--layerwise', action='store_true') # default = False
    parser.add_argument('--k-relwise', action='store_true')
    parser.add_argument('--before-topk', action='store_true')
    
    args = parser.parse_args()
    
    PERLIN_MODE = args.mode
    PERLIN_LAYERWISE = args.layerwise
    PERLIN_K_RELWISE = args.k_relwise
    PERLIN_BEFORE_TOPK = args.before_topk
    
    print(f"Perlin_trainer] perlin_mode: {PERLIN_MODE}")
    print(f"Perlin_trainer] perlin_layerwise: {PERLIN_LAYERWISE}")
    print(f"Perlin_trainer] perlin_k_relwise: {PERLIN_K_RELWISE}")
    print(f"Perlin_trainer] perlin_before_topk: {PERLIN_BEFORE_TOPK}")
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()