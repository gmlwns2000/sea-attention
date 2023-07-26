from dataclasses import asdict
import warnings

import torch
from torch import nn

from ..models import perlin_attention
from ..models import perlin_bert
from ..models import perlin_opt
from ..models.perlin_bert.compat import migrate_state_dict
from ..utils import seed
from .glue_trainer import Trainer as BaseGlueTrainer
from .glue_trainer import task_to_batch_size
from .lra_trainer import Trainer as BaseLraTrainer
from .opt_trainer import Trainer as BaseOptTrainer
from .opt_trainer import TrainerConfig as OptTrainerConfig

bool2int = lambda x: 1 if x else 0

def add_perlin_model_options(parser):
    parser.add_argument('--method', default='perlin', type=str)
    parser.add_argument('--layerwise', action='store_true', default=False)
    parser.add_argument('--enable-lora', action='store_true', default=False)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--k-colwise', action='store_true', default=False)
    parser.add_argument('--k-flatten-dim', default='batch', type=str)
    parser.add_argument('--attention-predictor-method', default='mlp', type=str)
    parser.add_argument('--performer-nb-feature-factor', default=1, type=float)
    parser.add_argument('--random-lookup', action='store_true', default=False)
    parser.add_argument('--random-lookup-count', default=3, type=int)
    parser.add_argument('--token-merging', action='store_true', default=False)
    parser.add_argument('--token-merging-preserve', default=0.2, type=float)
    parser.add_argument('--token-merging-ratio', default=0.5, type=float)
    return parser

def parse_perlin_model_options(args):
    kwargs = {
        'perlin_k':args.k,
        'attention_method':args.method,
        'perlin_k_flatten':not args.k_colwise,
        'perlin_k_flatten_dim': args.k_flatten_dim,
        'perlin_layerwise':args.layerwise,
        # NOTE HJ now lora is disable by default
        # 'perlin_lora':not args.disable_lora, 
        'perlin_lora':args.enable_lora,
        'perlin_attention_predictor_method':args.attention_predictor_method,
        'perlin_performer_nb_feature_factor':args.performer_nb_feature_factor,
        'perlin_random_lookup': args.random_lookup,
        'perlin_random_lookup_count': args.random_lookup_count,
        'perlin_token_merging': args.token_merging,
        'perlin_token_merging_preserve': args.token_merging_preserve,
        'perlin_token_merging_ratio': args.token_merging_ratio,
    }
    return kwargs

class BaseTrainer:
    def __init__(
        self,
        perlin_k = 7,
        perlin_k_flatten = True,
        perlin_k_flatten_dim = 'batch',
        perlin_layerwise = False,
        perlin_lora = True,
        attention_method = 'perlin',
        perlin_attention_predictor_method = 'mlp',
        perlin_performer_nb_feature_factor = 1,
        perlin_random_lookup = False,
        perlin_random_lookup_count = 3,
        perlin_token_merging = False,
        perlin_token_merging_preserve = 0.2,
        perlin_token_merging_ratio = 0.5,
        **kwargs,
    ) -> None:
        self.attention_method = attention_method
        self.perlin_k = perlin_k
        self.perlin_k_flatten = perlin_k_flatten
        self.perlin_k_flatten_dim = perlin_k_flatten_dim
        self.perlin_layerwise = perlin_layerwise
        self.perlin_lora = perlin_lora
        self.perlin_attention_predictor_method = perlin_attention_predictor_method
        self.perlin_performer_nb_feature_factor = perlin_performer_nb_feature_factor
        self.perlin_random_lookup = perlin_random_lookup
        self.perlin_random_lookup_count = perlin_random_lookup_count
        
        self.pelrin_performer_nb_feature_factor = perlin_performer_nb_feature_factor
        self.perlin_token_merging = perlin_token_merging
        self.perlin_token_merging_preserve = perlin_token_merging_preserve
        self.perlin_token_merging_ratio = perlin_token_merging_ratio
        
        # NOTE HJ default setting is defined in PerlinAttentionConfig dataclass
        self.perlin_config = perlin_attention.PerlinAttentionConfig(
            performer_nb_factor = perlin_performer_nb_feature_factor,
            k = perlin_k,
            k_flatten = perlin_k_flatten,
            k_flatten_dim = perlin_k_flatten_dim,
            random_lookup = perlin_random_lookup,
            random_lookup_count = perlin_random_lookup_count,
            attention_predictor_method = perlin_attention_predictor_method,
            layerwise = perlin_layerwise,
            lora_enabed = perlin_lora,
        )
        perlin_attention.register_default_config(self.perlin_config)
    
    def apply_model_options(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, perlin_bert.BertSelfAttention):
                module.attention_method = self.attention_method
                module.perlin_token_merging = self.perlin_token_merging
                module.perlin_token_merging_ratio = self.perlin_token_merging_ratio
                module.perlin_token_merging_preserve_ratio = self.perlin_token_merging_preserve
            elif isinstance(module, perlin_opt.OPTAttention):
                assert not self.perlin_token_merging, "opt does not support this!"
                module.attention_method = self.attention_method
        
        if self.perlin_layerwise:
            for name, param in model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for module in model.modules():
                if isinstance(module, perlin_attention.PerlinSelfAttention):
                    if not self.perlin_lora: # activate QKV
                        for p in module.parameters():
                            p.requires_grad = True
        return model

    def format_exp(self, name: str):
        name_k_window_size = f'_k{self.perlin_k}' if self.perlin_k != 7 else ''
        name_k_flatten_dim = f'_kdim_{self.perlin_k_flatten_dim}' if self.perlin_k_flatten_dim != 'batch' else ''
        name_lora = '_full' if not self.perlin_lora else ''
        name_predictor = f'_pred{self.perlin_attention_predictor_method}' if self.perlin_attention_predictor_method != 'mlp' else ''
        name_nbf = f'_nbf{self.perlin_performer_nb_feature_factor}' if self.perlin_performer_nb_feature_factor != 1 else ''
        name_random_lookup = f'_rl_c{self.perlin_random_lookup_count}' if self.perlin_random_lookup else ''
        name_tome = f'_tome_r{self.perlin_token_merging_ratio}_p{self.perlin_token_merging_preserve}' if self.perlin_token_merging else ''
        name = f'{name}'\
            f'_kf{bool2int(self.perlin_k_flatten)}'\
            f'_lw{bool2int(self.perlin_layerwise)}'\
            f'_{self.attention_method}{name_k_window_size}{name_lora}{name_predictor}{name_nbf}{name_random_lookup}{name_tome}{name_k_flatten_dim}'
        return name

    def get_global_config(self):
        return asdict(perlin_attention.get_default_config())

class GlueTrainer(BaseGlueTrainer, BaseTrainer):
    def __init__(
        self, 
        subset = 'mnli',
        lr = 1e-5,
        epochs = 20,
        disable_amp = False,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        batch_size = None,
        **kwargs,
    ):
        BaseTrainer.__init__(self, **kwargs)
        
        # NOTE HJ origimal batch sizes
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
        
        if batch_size is not None:
            task_to_batch_size[subset] = batch_size
        
        BaseGlueTrainer.__init__(
            self,
            subset=subset,
            model_cls=perlin_bert.BertForSequenceClassification,
            amp_enabled=not disable_amp,
            trainer_name=self.format_exp('glue' if subset == 'mnli' else f'glue_{subset}'),
            using_kd=not self.perlin_layerwise,
            using_loss=not self.perlin_layerwise,
            eval_steps=2000,
            lr = lr,
            epochs = epochs,
            gradient_checkpointing = gradient_checkpointing,
            gradient_accumulation_steps = gradient_accumulation_steps,
            high_lr_names=['perlin'],
            wandb_configs=self.get_global_config(),
        )
        
        self.apply_model_options(self.model)
    
    def migrate_state_dict(self, state_dict):
        return migrate_state_dict(state_dict)

class LraTrainer(BaseLraTrainer, BaseTrainer):
    def __init__(
        self, 
        subset: str = 'listops',
        disable_amp: bool = False,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        **kwargs
    ):
        BaseTrainer.__init__(self, **kwargs)
        
        warnings.warn(f'epochs({kwargs.get("epochs", 20)}) are ignored')
        
        BaseLraTrainer.__init__(
            self,
            exp_name=self.format_exp(f'lra_{subset}'),
            subset=subset,
            model_cls=perlin_bert.BertForSequenceClassification,
            gradient_checkpointing = gradient_checkpointing,
            gradient_accumulation_steps = gradient_accumulation_steps,
            using_kd=True,
            amp_enabled=not disable_amp,
        )
        
        self.apply_model_options(self.model)
    
    def migrate_state_dict(self, state_dict):
        return migrate_state_dict(state_dict)

class OptTrainer(BaseOptTrainer, BaseTrainer):
    def __init__(
        self, 
        model: str = 'opt',
        subset: str = 'wikitext2',
        disable_amp: bool = False,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        epochs: int = None,
        max_seq_len: int = None,
        **kwargs
    ):
        BaseTrainer.__init__(self, **kwargs)
        
        model = {
            'opt': 'opt-125m',
        }.get(model, model)
        
        model_config = {
            'opt-125m': {
                'wikitext2': 'Aalaa/opt-125m-wikitext2'
            }
        }[model][subset]
        
        BaseOptTrainer.__init__(self, OptTrainerConfig(
            experiment_name=self.format_exp(f'{model}_{subset}'),
            model_cls=perlin_opt.OPTForCausalLM,
            model_config=model_config,
            amp_enabled=not disable_amp,
            epochs=epochs if epochs is not None else 100,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            max_seq_len=max_seq_len,
        ))
        
        self.apply_model_options(self.model)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='glue', type=str)
    parser.add_argument('--model', default='bert', type=str)
    parser.add_argument('--subset', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--max-seq-len', default=32000, type=int)
    parser.add_argument('--load-checkpoint', default=None, type=str)
    parser.add_argument('--load-only-additionals', action='store_true')
    
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', default=False)
    
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    
    seed()
    
    print(args)
    
    if args.subset is None:
        if args.dataset == 'glue':
            args.subset = 'mnli'
        elif args.dataset == 'lra':
            args.subset = 'listops'
        elif args.dataset == 'wikitext2':
            args.subset = 'wikitext2'
        else:
            raise Exception()

    if args.dataset == 'glue':
        assert args.model in ['bert']
    elif args.dataset == 'lra':
        assert args.model in ['bert']
    elif args.dataset == 'wikitext2':
        assert args.model in ['opt']
    else:
        raise Exception()
    
    kwargs = {
        'model': args.model,
        'subset':args.subset,
        'epochs': args.epochs,
        'gradient_checkpointing':args.gradient_checkpointing,
        'gradient_accumulation_steps':args.gradient_accumulation_steps,
        'disable_amp': args.disable_amp,
        'max_seq_len': args.max_seq_len,
    }
    kwargs.update(parse_perlin_model_options(args))
    
    if args.dataset == 'glue':
        trainer = GlueTrainer(**kwargs)
    elif args.dataset == 'lra':
        trainer = LraTrainer(**kwargs)
    elif args.model in ['opt']:
        trainer = OptTrainer(**kwargs)
    else:
        raise Exception()
    
    if args.load_checkpoint is not None:
        if args.load_checkpoint in ['auto', 'defualt', '']: 
            trainer.load()
        else:
            trainer.load(args.load_checkpoint)
    
    if args.load_only_additionals:
        trainer.load_state_from_base()

    trainer.main()