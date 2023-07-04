import warnings
import torch
from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

bool2int = lambda x: 1 if x else 0

class Trainer(BaseTrainer):
    def __init__(
        self, 
        subset = 'mnli',
        lr = 1e-4,
        epochs = 20,
        perlin_k = 7,
        perlin_k_flatten = True,
        perlin_layerwise = False,
        perlin_lora = True,
        attention_method = 'perlin',
        perlin_attention_predictor_method = 'mlp',
        perlin_performer_nb_feature_factor = 1,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
    ):
        self.attention_method = attention_method
        self.perlin_k = perlin_k
        self.perlin_k_flatten = perlin_k_flatten
        self.perlin_layerwise = perlin_layerwise
        self.perlin_lora = perlin_lora
        self.perlin_attention_predictor_method = perlin_attention_predictor_method
        self.perlin_performer_nb_feature_factor = perlin_performer_nb_feature_factor
        perlin.PERLIN_PERFORMER_NB_FACTOR = perlin_performer_nb_feature_factor
        
        task_to_batch_size['mnli'] = (16 if not perlin_layerwise else 24) // gradient_accumulation_steps
        
        name_k_window_size = f'_k{perlin_k}' if perlin_k != 7 else ''
        name_lora = '_full' if not perlin_lora else ''
        name = f'perlin_trainer'\
            f'_kf{bool2int(perlin_k_flatten)}'\
            f'_lw{bool2int(perlin_layerwise)}'\
            f'_{attention_method}{name_k_window_size}{name_lora}'
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=True,
            trainer_name=name,
            using_kd=(not perlin_layerwise),
            using_loss=not perlin_layerwise,
            eval_steps=2000,
            lr = lr,
            epochs = epochs,
            gradient_checkpointing = gradient_checkpointing,
            gradient_accumulation_steps = gradient_accumulation_steps,
            high_lr_names=['perlin'],
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.attention_method = attention_method
                module.perlin_k_flatten = perlin_k_flatten
                module.perlin_k = perlin_k
                module.perlin_attention_predictor_method = perlin_attention_predictor_method
        
        if perlin_layerwise:
            for name, param in self.model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for module in self.model.modules():
                if isinstance(module, perlin.BertSelfAttention):
                    module.perlin_layerwise = True
                    module.perlin_lora_enabled = perlin_lora
                    if not perlin_lora: # activate QKV
                        for p in module.parameters():
                            p.requires_grad = True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subset', default='mnli', type=str)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    
    parser.add_argument('--method', default='perlin', type=str)
    parser.add_argument('--layerwise', action='store_true', default=False)
    parser.add_argument('--disable-lora', action='store_true', default=False)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--k-colwise', action='store_true', default=False)
    parser.add_argument('--attention-predictor-method', default='mlp', type=str)
    parser.add_argument('--performer-nb-feature-factor', default=1, type=float)
    args = parser.parse_args()
    
    trainer = Trainer(
        subset=args.subset,
        perlin_k=args.k,
        attention_method=args.method,
        perlin_k_flatten=not args.k_colwise,
        perlin_layerwise=args.layerwise,
        perlin_lora=not args.disable_lora,
        perlin_attention_predictor_method=args.attention_predictor_method,
        perlin_performer_nb_feature_factor=args.performer_nb_feature_factor,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.main()