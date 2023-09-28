import random
import time
import transformers, os
import torch
from torch import nn, optim
import tqdm
import wandb
from ..models import hf_bert as berts
# from ..dataset.lra_benchmarks_ import get_loaders
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to
from ..dataset.lra_benchmarks.list_ops import get_tokenizer as get_tokenizer_listops
from ..dataset.lra_benchmarks.text import get_tokenizer as get_tokenizer_text
from ..dataset.lra_benchmarks.image import get_tokenizer as get_tokenizer_image
from ..utils import Metric, seed

from xformers.benchmarks.LRA.code.dataset import LRADataset

BF16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LRA_TASKS_INITIAL = {
    'listops': {
        'batch_size': 32,
        'dataloader_fn': lambda bs: get_loaders('listops', bs),
        'lr': 2e-3,
        'wd': 1e-1,
        'epochs': 30,
        'eval_steps': 6000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 8,
        'config': berts.BertConfig(
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            num_labels=10,
            vocab_size=get_tokenizer_listops().vocab_size,
        )
    },
    'text': {
        'batch_size': 16,
        'dataloader_fn': lambda bs: get_loaders('text', bs),
        'lr': 1e-5,
        'wd': 1e-1,
        'epochs': 30,
        'eval_steps': 12000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 2,
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=4,
            num_hidden_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_labels=2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            vocab_size=get_tokenizer_text().vocab_size,
        )
    },
    'image': {
        'batch_size': 256,
        'dataloader_fn': lambda bs: get_loaders('image', bs),
        'lr': 1e-3,
        'wd': 0.0,
        'epochs': 500,
        'eval_steps': 12000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 256//256,
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=1,
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=64,
            num_labels=10,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.2,
            vocab_size=get_tokenizer_image().vocab_size,
        )
    }
}

LRA_TASKS = {
    "text": {
        'wandb_steps': 10,
        'epochs': 30,
        "dataloader_fn": lambda bs: get_loaders('text', bs),
        "dataset": {
            "train": 25000,
            "dev": 25000,
            "test": 25000
        },
        "training": {
            "mixed_precision": false,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 8000,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 500,
            "num_train_steps": 20000,
            "num_eval_steps": 779,
            "gradient_accumulation": 1
        },
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=4,
            num_hidden_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_labels=2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            vocab_size=512 # get_tokenizer_text().vocab_size,
        )
        "model": {
            "pooling_mode": "cls",
            "common": {
                "dim_model": 256,
                "num_classes": 2,
                "seq_len": 4096,
                "num_heads": 4,
                "vocab_size": 512,
                "dropout": 0.1
            },
            "xformer": [
                {
                    "reversible": false,
                    "block_type": "encoder",
                    "num_layers": 4,
                    "residual_norm_style": "pre",
                    "position_encoding_config": {
                        "name": "vocab"
                    },
                    "multi_head_config": {
                        "residual_dropout": 0.1,
                        "attention": {
                            "name": "generic",
                            "causal": false
                        }
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "activation": "gelu",
                        "hidden_layer_multiplier": 4
                    }
                }
            ]
        },
        "extra_settings": {
            "attention": {
                "favor": {
                    "dim_features": 256,
                    "iter_before_redraw": 1000
                },
                "nystrom": {
                    "conv_kernel_size": 35,
                    "num_landmarks": 128
                }
            }
        }
    },
    "listops": {
        'wandb_steps': 10,
        'epochs': 30,
        "dataloader_fn": lambda bs: get_loaders('listops', bs),
        "dataset": {
            "train": 96000,
            "dev": 2000,
            "test": 2000
        },
        "training": {
            "mixed_precision": false,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 1000,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 50,
            "num_train_steps": 10000,
            "num_eval_steps": 62,
            "gradient_accumulation": 2
        },
        'config': berts.BertConfig(
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            num_labels=10,
            vocab_size=32, # get_tokenizer_listops().vocab_size
        ),
        "model": {
            "pooling_mode": "mean",
            "common": {
                "num_classes": 10,
                "num_heads": 2,
                "dim_model": 64,
                "seq_len": 2048,
                "vocab_size": 32,
                "dropout": 0.1
            },
            "xformer": [
                {
                    "reversible": false,
                    "block_type": "encoder",
                    "num_layers": 2,
                    "residual_norm_style": "pre",
                    "position_encoding_config": {
                        "name": "vocab"
                    },
                    "multi_head_config": {
                        "residual_dropout": 0.1,
                        "attention": {
                            "name": "generic",
                            "causal": false
                        }
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "activation": "gelu",
                        "hidden_layer_multiplier": 2
                    }
                }
            ],
            "extra_settings": {
                "attention": {
                    "favor": {
                        "dim_features": 256,
                        "iter_before_redraw": 1000
                    },
                    "nystrom": {
                        "conv_kernel_size": 35,
                        "num_landmarks": 128
                    }
                }
            }
        }
    },
    "retrieval": {
        "dataloader_fn": lambda bs: get_loaders('retrieval', bs),
        "dataset": {
            "train": 147086,
            "dev": 18090,
            "test": 17437
        },
        "training": {
            "mixed_precision": false,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 800,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 300,
            "num_train_steps": 30000,
            "num_eval_steps": 565,
            "gradient_accumulation": 2
        },
        "model": {
            "pooling_mode": "mean",
            "common": {
                "num_classes": 2,
                "num_heads": 2,
                "seq_len": 4096,
                "dim_model": 64,
                "vocab_size": 512,
                "dropout": 0.1
            },
            "xformer": [
                {
                    "reversible": true,
                    "block_type": "encoder",
                    "num_layers": 2,
                    "residual_norm_style": "pre",
                    "position_encoding_config": {
                        "name": "vocab"
                    },
                    "multi_head_config": {
                        "residual_dropout": 0.1,
                        "attention": {
                            "name": "generic",
                            "causal": false
                        }
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "activation": "gelu",
                        "hidden_layer_multiplier": 2
                    }
                }
            ],
            "extra_settings": {
                "attention": {
                    "favor": {
                        "dim_features": 256,
                        "iter_before_redraw": 1000
                    },
                    "nystrom": {
                        "conv_kernel_size": 35,
                        "num_landmarks": 128
                    }
                }
            }
        }
    },
    "pathfinder32": {
        "training": {
            "mixed_precision": false,
            "batch_size": 256,
            "learning_rate": 0.0001,
            "warmup": 312,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 312,
            "num_train_steps": 62400,
            "num_eval_steps": 312,
            "gradient_accumulation": 1
        },
        "model": {
            "pooling_mode": "mean",
            "common": {
                "vocab_size": 512,
                "num_classes": 2,
                "num_heads": 2,
                "seq_len": 1024,
                "dim_model": 64,
                "dropout": 0.1
            },
            "xformer": [
                {
                    "reversible": true,
                    "block_type": "encoder",
                    "num_layers": 2,
                    "residual_norm_style": "pre",
                    "position_encoding_config": {
                        "name": "vocab"
                    },
                    "multi_head_config": {
                        "residual_dropout": 0.1,
                        "attention": {
                            "name": "generic",
                            "causal": false
                        }
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "activation": "gelu",
                        "hidden_layer_multiplier": 2
                    }
                }
            ],
            "extra_settings": {
                "attention": {
                    "favor": {
                        "dim_features": 256,
                        "iter_before_redraw": 1000
                    },
                    "nystrom": {
                        "conv_kernel_size": 35,
                        "num_landmarks": 128
                    }
                }
            }
        }
    },
    "image": {
        'wandb_steps': 10,
        'epochs': 500,
        "dataloader_fn": lambda bs: get_loaders('image', bs),
        "dataset": {
            "train": 45000,
            "dev": 5000,
            "test": 10000
        },
        "training": {
            "mixed_precision": false,
            "batch_size": 256,
            "learning_rate": 0.0001,
            "warmup": 175,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 175,
            "num_train_steps": 35000,
            "num_eval_steps": 20,
            "gradient_accumulation": 1
        },
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=1,
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=64,
            num_labels=10,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.2,
            vocab_size=512, # get_tokenizer_image().vocab_size,
        ),
        "model": {
            "pooling_mode": "mean",
            "common": {
                "vocab_size": 512,
                "num_classes": 10,
                "num_heads": 2,
                "seq_len": 1024,
                "dim_model": 64,
                "dropout": 0.1
            },
            "xformer": [
                {
                    "reversible": true,
                    "block_type": "encoder",
                    "num_layers": 2,
                    "residual_norm_style": "pre",
                    "position_encoding_config": {
                        "name": "vocab"
                    },
                    "multi_head_config": {
                        "residual_dropout": 0.1,
                        "attention": {
                            "name": "generic",
                            "causal": false
                        }
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "activation": "gelu",
                        "hidden_layer_multiplier": 2
                    }
                }
            ],
            "extra_settings": {
                "attention": {
                    "favor": {
                        "dim_features": 256,
                        "iter_before_redraw": 1000
                    },
                    "nystrom": {
                        "conv_kernel_size": 35,
                        "num_landmarks": 128
                    }
                }
            }
        }
    }
}

def build_dataloaders(
    task,
    config_training: Dict,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    world_size = 1 # TODO
    datasets = {}
    for component in ("train", "dev", "test"):
        datasets[component] = LRADataset(
            file_path=f"datasets/{task}.{component}.pickle",
            seq_len=config_training["seq_len"],
        )

    # Gradient accumulation
    accumu_steps = config_training["gradient_accumulation"]
    logging.info(f"accumu_steps={accumu_steps}")

    # Batch size
    per_gpu_batch_size = (
        config_training["batch_size"] // world_size // accumu_steps
    )
    logging.warning(
        f"Requested batch size: {config_training['batch_size']}. Given world\
            size and grad accumulation, per-gpu batch is\
            {per_gpu_batch_size}"
    )

    dataloaders = {
        k: DataLoader(
            v,
            batch_size=per_gpu_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        for k, v in datasets.items()
    }
    return dataloaders

class Trainer:
    def __init__(
        self,
        exp_name: str = 'listops',
        subset: str = 'listops',
        
        model_cls: berts.BertForSequenceClassification = berts.BertForSequenceClassification,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        using_kd: bool = False,
        kd_checkpoint: str = None,
        
        amp_enabled: bool = True,
        device: int = 0,
    ) -> None:
        seed()
        
        task_desc = LRA_TASKS[subset]
        
        self.exp_name = exp_name
        self.subset = subset
        self.batch_size = task_desc['training']['batch_size']
        self.epochs = task_desc['epochs']
        # self.epochs = task_desc['epochs'] TODO
        self.lr = task_desc['training']['learning_rate']
        self.wd = task_desc['training']['weight_decay']
        self.eval_steps = task_desc['training']['num_eval_steps']
        self.wandb_steps = task_desc['wandb_steps'] # TODO
        # self.wandb_steps = task_desc['wandb_steps'] # TODO
        self.device = device
        self.amp_enabled = amp_enabled
        self.gradient_checkpointing = gradient_checkpointing
        assert not gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps * task_desc['training']['gradient_accumulation']
        assert (self.batch_size % gradient_accumulation_steps) == 0
        self.batch_size = self.batch_size // self.gradient_accumulation_steps
        
        self.using_kd = using_kd
        self.kd_checkpoint = kd_checkpoint
        if self.kd_checkpoint is None:
            self.kd_checkpoint = f'./saves/trainer/lra_trainer/{subset}/checkpoint.pth'
        
        config_traing = task_desc['training']
        dataloaders = build_dataloaders(subset, config_training)
        self.train_loader = dataloaders["train"]
        self.test_loader = dataloaders["test"]

        # self.train_loader, self.test_loader = task_desc['dataloader_fn'](self.batch_size)
        
        self.model = model_cls(task_desc['config'])
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.epoch = 0
        self.step = 0
        
        self.base_model = None
        if self.using_kd:
            state = torch.load(self.kd_checkpoint, map_location='cpu')
            self.base_model = berts.BertForSequenceClassification(self.model.config)
            self.base_model.load_state_dict(state['model'])
            self.base_model.to(self.device)
            del state
            print('loaded base model from', self.kd_checkpoint)
    
    def train_step(self, batch):
        base_model = self.base_model
        model = self.model
        
        model.train()
        if base_model is not None: base_model.eval()
        
        with torch.autocast('cuda', BF16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            if self.using_kd:
                with torch.no_grad():
                    output_teacher = base_model(**batch)
                batch['teacher'] = base_model
            output = model(**batch)
            loss = output.loss
        
        loss_details = {'loss': loss, 'loss_model': loss}
        
        if self.using_kd:
            loss_model = loss * 0.1
            
            loss_kd = 0
            for ilayer in range(len(output_teacher.hidden_states)):
                loss_kd += torch.nn.functional.mse_loss(
                    output_teacher.hidden_states[ilayer], 
                    output.hidden_states[ilayer]
                )
            loss_kd = loss_kd / len(output_teacher.hidden_states) * 10
            assert len(output_teacher.hidden_states) > 0
            
            loss_special = 0
            if hasattr(self.model, 'calc_loss_special'):
                # warnings.warn('special loss found!')
                loss_special = self.model.calc_loss_special()
            
            loss = loss_model + loss_kd + loss_special
            
            loss_details['loss'] = loss.item()
            loss_details['loss_model'] = loss_model.item()
            loss_details['loss_kd'] = loss_kd.item()
            loss_details['loss_sp'] = loss_special.item()
        
        self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        
        if ((self.step + 1) % self.gradient_accumulation_steps) == 0:
            self.scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.scaler.update()
        
        self.step += 1
        
        return loss, loss_details
    
    def train_epochs(self):
        m = Metric()
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                loss, loss_details = self.train_step(batch)
                
                if (self.step % self.eval_steps) == 0:
                    metric = self.evaluate()
                    self.save()
                    m = Metric()
                    wandb.log({'eval/metric': metric}, step=self.step)
                
                if (self.step % self.wandb_steps) == 0:
                    wandb_dict = {f'train/{k}': v for k, v in loss_details.items()}
                    wandb_dict['train/epoch'] = istep / len(pbar) + self.epoch
                    wandb.log(wandb_dict, step=self.step)
                
                pbar.set_description((
                    f'[{self.epoch}/{self.epochs}] '
                    f'L:{m.update(loss.item(), "l"):.4f}({m.update(loss_details.get("loss_model", 0.0), "lm"):.4f}) '
                    f'Lsp:{m.update(loss_details.get("loss_sp", 0.0), "lsp"):.4f} '
                    f'Lkd:{m.update(loss_details.get("loss_kd", 0.0), "lkd"):.4f}'
                ).strip())
    
    def evaluate(self):
        model = self.model
        base_model = self.base_model
        model.eval()
        if base_model is not None: base_model.eval()
        
        acc_sum = acc_count = 0
        for batch in tqdm.tqdm(self.test_loader, dynamic_ncols=True):
            batch = batch_to(batch, self.device)
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            with torch.no_grad(), torch.autocast('cuda', BF16, enabled=self.amp_enabled):
                if base_model is not None:
                    base_model(**batch)
                    batch['teacher'] = base_model
                output = model(**batch)
                logits = output.logits
            acc = ((torch.argmax(logits, dim=-1) == batch['labels'])*1.0).sum()
            acc_sum += acc.item()
            acc_count += len(batch['input_ids'])
        
        acc_sum = acc_sum / (acc_count + 1e-8)
        print('accuracy:', acc_sum)
        
        return acc_sum
    
    def checkpoint_path(self):
        dir = f'./saves/trainer/lra_trainer/{self.exp_name}'
        os.makedirs(dir, exist_ok=True)
        return f'{dir}/checkpoint.pth'
    
    def save(self, path=None):
        if path is None: path = self.checkpoint_path()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)
        print('saved', path)
    
    def load(self, path=None):
        if path is None: path = self.checkpoint_path()
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        try:
            self.optimizer.load_state_dict(state['optimizer'])
            self.scaler.load_state_dict(state['scaler'])
        except Exception as ex:
            print('except while load', ex)
        del state
        print('loaded', path)
    
    def main(self):
        from ..utils.secrets import WANDB_KEY, USER_NAME
        os.environ['WANDB_API_KEY'] = WANDB_KEY
        wandb.init(
            project=f"[{USER_NAME}] perlin-lra",
            name=f"{self.exp_name}-{int(time.time()*1000 % 1000)}",
            config={
                "lr": self.lr,
                "subset": self.subset,
                "epochs": self.epochs,
            }
        )
        wandb.watch(self.model, log='all')
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            self.train_epochs()
            metric = self.evaluate()
            wandb.log({'eval/metric': metric}, step=self.step)
            self.save()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', default='listops', type=str)
    args = parser.parse_args()
    
    t = Trainer(
        subset=args.subset,
        exp_name=args.subset,
    )
    t.main()