import warnings
import torch
from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

PERLIN_K_FLATTEN = True
PERLIN_LAYERWISE = False
PERLIN_MODE = 'perlin'

bool2int = lambda x: 1 if x else 0

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        global PERLIN_LAYERWISE, PERLIN_MODE, PERLIN_K_FLATTEN

        task_to_batch_size['mnli'] = 16 if not PERLIN_LAYERWISE else 24

        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=False,
            trainer_name=f'perlin_trainer_kf{bool2int(PERLIN_K_FLATTEN)}_lw{bool2int(PERLIN_LAYERWISE)}_{PERLIN_MODE}',
            using_kd=(not PERLIN_LAYERWISE) and (PERLIN_MODE != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps=2000,
            lr = 1e-4,
            epochs = 20
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.perlin_mode = PERLIN_MODE
                module.perlin_k_flatten = PERLIN_K_FLATTEN
        
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
    parser.add_argument('--layerwise', action='store_true', default=False)
    parser.add_argument('--k-colwise', action='store_true', default=False)
    args = parser.parse_args()
    
    PERLIN_MODE = args.mode
    PERLIN_K_FLATTEN = not args.k_colwise
    PERLIN_LAYERWISE = args.layerwise
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()