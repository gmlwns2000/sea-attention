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
        # NOTE why global?
        global ATTENTION_METHOD, PERLIN_LAYERWISE, PERLIN_K_RELWISE, PERLIN_REDRAW_PROJ # PERLIN_CHECKOUT_DENSE_ATTENTION

        task_to_batch_size['mnli'] = 16 if not PERLIN_LAYERWISE else 24
        
        self.attention_method = ATTENTION_METHOD
        self.perlin_k_relwise = PERLIN_K_RELWISE
        # self.perlin_checkout_dense_attention = PERLIN_CHECKOUT_DENSE_ATTENTION
        self.perlin_redraw_proj = PERLIN_REDRAW_PROJ
        
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=True,
            trainer_name=f'perlin_trainer_lw{bool2int(PERLIN_LAYERWISE)}_rw{bool2int(PERLIN_K_RELWISE)}_rdp{bool2int(PERLIN_REDRAW_PROJ)}_{ATTENTION_METHOD}',
            using_kd=(not PERLIN_LAYERWISE) and (ATTENTION_METHOD != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps = 2000, 
            lr = 1e-4,
            epochs = 20,
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.attention_method = ATTENTION_METHOD
                module.perlin_k_relwise = PERLIN_K_RELWISE
                module.perlin_redraw_proj = PERLIN_REDRAW_PROJ
                # breakpoint()
        
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
            # breakpoint()
                    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='perlin', type=str)
    parser.add_argument('--layerwise', action='store_true') # default = False
    parser.add_argument('--k-relwise', action='store_true')
    parser.add_argument('--redraw-proj', action='store_true')
    
    args = parser.parse_args()
    
    ATTENTION_METHOD = args.method
    PERLIN_LAYERWISE = args.layerwise
    PERLIN_K_RELWISE = args.k_relwise
    PERLIN_REDRAW_PROJ = args.redraw_proj
    
    print(f"Perlin_trainer] attention_method: {ATTENTION_METHOD}")
    print(f"Perlin_trainer] perlin_layerwise: {PERLIN_LAYERWISE}")
    print(f"Perlin_trainer] perlin_k_relwise: {PERLIN_K_RELWISE}")
    print(f"Perlin_trainer] perlin_redraw_proj: {PERLIN_REDRAW_PROJ}")
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()