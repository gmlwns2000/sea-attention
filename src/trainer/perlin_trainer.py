import warnings
import torch
from torch import nn
from .bert_glue_trainer import Trainer as BaseGlueTrainer
from .lra_trainer import Trainer as BaseLraTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

bool2int = lambda x: 1 if x else 0

class BaseTrainer:
    def __init__(
        self,
        perlin_k = 7,
        perlin_k_flatten = True,
        perlin_layerwise = False,
        perlin_lora = True,
        attention_method = 'perlin',
        perlin_attention_predictor_method = 'mlp',
        perlin_performer_nb_feature_factor = 1,
        **kwargs,
    ) -> None:
        self.attention_method = attention_method
        self.perlin_k = perlin_k
        self.perlin_k_flatten = perlin_k_flatten
        self.perlin_layerwise = perlin_layerwise
        self.perlin_lora = perlin_lora
        self.perlin_attention_predictor_method = perlin_attention_predictor_method
        self.perlin_performer_nb_feature_factor = perlin_performer_nb_feature_factor
        perlin.PERLIN_PERFORMER_NB_FACTOR = perlin_performer_nb_feature_factor
    
    def apply_model_options(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.attention_method = self.attention_method
                module.perlin_k_flatten = self.perlin_k_flatten
                module.perlin_k = self.perlin_k
                module.perlin_attention_predictor_method = self.perlin_attention_predictor_method
        
        if self.perlin_layerwise:
            for name, param in model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for module in model.modules():
                if isinstance(module, perlin.BertSelfAttention):
                    module.perlin_layerwise = True
                    module.perlin_lora_enabled = self.perlin_lora
                    if not self.perlin_lora: # activate QKV
                        for p in module.parameters():
                            p.requires_grad = True
        return model

    def format_exp(self, name: str):
        name_k_window_size = f'_k{self.perlin_k}' if self.perlin_k != 7 else ''
        name_lora = '_full' if not self.perlin_lora else ''
        name = f'{name}'\
            f'_kf{bool2int(self.perlin_k_flatten)}'\
            f'_lw{bool2int(self.perlin_layerwise)}'\
            f'_{self.attention_method}{name_k_window_size}{name_lora}'
        return name

class GlueTrainer(BaseGlueTrainer, BaseTrainer):
    def __init__(
        self, 
        subset = 'mnli',
        lr = 1e-5,
        epochs = 20,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        **kwargs,
    ):
        BaseTrainer.__init__(self, **kwargs)
        
        # task_to_batch_size = {
        #     "cola": 64,
        #     "mnli": 4,
        #     "mrpc": 32,
        #     "qnli": 4,
        #     "qqp":  16,
        #     "rte":  8,
        #     "sst2": 16,
        #     "stsb": 16,
        #     "wnli": 32,
        #     "bert": 4,
        # }
        task_to_batch_size['cola'] = (64 if not self.perlin_layerwise else 128) // gradient_accumulation_steps
        task_to_batch_size['mnli'] = (16 if not self.perlin_layerwise else 24) // gradient_accumulation_steps
        task_to_batch_size['mrpc'] = (64 if not self.perlin_layerwise else 96) // gradient_accumulation_steps
        task_to_batch_size['qnli'] = (16 if not self.perlin_layerwise else 24) // gradient_accumulation_steps
        task_to_batch_size['qqp']  = (64 if not self.perlin_layerwise else 96) // gradient_accumulation_steps
        task_to_batch_size['rte']  = (32 if not self.perlin_layerwise else 48) // gradient_accumulation_steps
        task_to_batch_size['sst2'] = (64 if not self.perlin_layerwise else 96) // gradient_accumulation_steps
        task_to_batch_size['stsb'] = (64 if not self.perlin_layerwise else 96) // gradient_accumulation_steps
        task_to_batch_size['wnli'] = (64 if not self.perlin_layerwise else 96) // gradient_accumulation_steps
        
        BaseGlueTrainer.__init__(
            self,
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=True,
            trainer_name=self.format_exp('glue' if subset == 'mnli' else f'glue_{subset}'),
            using_kd=not self.perlin_layerwise,
            using_loss=not self.perlin_layerwise,
            eval_steps=2000,
            lr = lr,
            epochs = epochs,
            gradient_checkpointing = gradient_checkpointing,
            gradient_accumulation_steps = gradient_accumulation_steps,
            high_lr_names=['perlin'],
        )
        
        self.apply_model_options(self.model)

class LraTrainer(BaseLraTrainer, BaseTrainer):
    def __init__(
        self, 
        subset: str = 'listops',
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        **kwargs
    ):
        BaseTrainer.__init__(self, **kwargs)
        
        BaseLraTrainer.__init__(
            self,
            exp_name=self.format_exp(f'lra_{subset}'),
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            gradient_checkpointing = gradient_checkpointing,
            gradient_accumulation_steps = gradient_accumulation_steps,
            using_kd=True,
        )
        
        self.apply_model_options(self.model)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='glue', type=str)
    parser.add_argument('--subset', default=None, type=str)
    
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
    
    if args.subset is None:
        if args.dataset == 'glue':
            args.subset = 'mnli'
        elif args.dataset == 'lra':
            args.subset = 'listops'
        else:
            raise Exception()
    
    kwargs = {
        'subset':args.subset,
        'perlin_k':args.k,
        'attention_method':args.method,
        'perlin_k_flatten':not args.k_colwise,
        'perlin_layerwise':args.layerwise,
        'perlin_lora':not args.disable_lora,
        'perlin_attention_predictor_method':args.attention_predictor_method,
        'perlin_performer_nb_feature_factor':args.performer_nb_feature_factor,
        'gradient_checkpointing':args.gradient_checkpointing,
        'gradient_accumulation_steps':args.gradient_accumulation_steps,
    }
    
    if args.dataset == 'glue':
        trainer = GlueTrainer(**kwargs)
    elif args.dataset == 'lra':
        trainer = LraTrainer(**kwargs)
    else:
        raise Exception()

    trainer.main()