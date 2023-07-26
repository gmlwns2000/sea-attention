from dataclasses import dataclass, field, asdict

import tqdm
from ..utils import seed, batch_to, Metric
from ..models import hf_opt as opt
from typing import List, Dict, Tuple
import torch
import wandb
import transformers
from torch import nn, optim
import os
from ..dataset.wikitext2 import get_dataloader
import gc

@dataclass
class TrainerConfig:
    # trainer metadata
    experiment_name: str = 'opt_wikitext2'
    eval_steps: int = 2000
    wandb_steps: int = 20
    
    # optimization flags
    # TODO grad checkpointing is not correct...
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 8
    amp_enabled: bool = True
    
    # experiment settings
    dataset: str = 'wikitext2'
    teacher_model_cls: opt.OPTForCausalLM = opt.OPTForCausalLM
    model_cls: opt.OPTForCausalLM = opt.OPTForCausalLM
    model_config: str = 'Aalaa/opt-125m-wikitext2'
    # model_config: str = 'lnair/opt-350m-wikitext2'
    lr: float = 1e-5
    wd: float = 1e-2
    epochs: int = 100
    batch_size: int = 1
    load_ignore_keys: List[str] = field(default_factory=lambda: ['perlin'])
    high_lr_names: List[str] = field(default_factory=lambda: ['perlin'])
    using_kd: bool = True
    using_loss: bool = True
    # NOTE decrease this only for DEBUG!!, this should be larger than 2048 on OPT
    max_seq_len: int = 32000
    
BF_16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class Trainer:
    def __init__(self, config: TrainerConfig = None) -> None:
        seed()
        
        self.config = config if config is not None else TrainerConfig()
        self.device = 0
        
        self.init_model()
        self.init_loader()
        self.init_optimizer()
    
    def init_model(self):
        teacher = self.config.teacher_model_cls.from_pretrained(
            self.config.model_config
        ).eval()
        
        student = self.config.model_cls(teacher.config)
        try:
            missing_keys, unexpected_keys = student.load_state_dict(teacher.state_dict(), strict=False)
            missing_keys = [k for k in missing_keys if not any([s in k for s in self.config.load_ignore_keys])]
            unexpected_keys = [k for k in unexpected_keys if not any([s in k for s in self.config.load_ignore_keys])]
            if len(missing_keys) > 0: 
                print('during init model, missing keys are:', missing_keys)
            if len(unexpected_keys) > 0: 
                print('during init model, unexpected keys are:', unexpected_keys)
        except Exception as ex:
            print(ex)
        
        self.base_model = teacher.to(self.device)
        self.model = student.to(self.device)
        
        # compatible with GPT2
        if hasattr(self.model.config, 'n_positions'):
            max_length = self.model.config.n_positions
        else:
            max_length = self.model.config.max_position_embeddings
        self.max_seq_len = min(self.config.max_seq_len, max_length)
        
        if self.config.gradient_checkpointing:
            print('patch gradient checkpointing')
            for m in self.model.modules():
                if hasattr(m, 'gradient_checkpointing'):
                    m.gradient_checkpointing = True
                    m.config.use_cache = False
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_config)
    
    def init_loader(self):
        if self.config.dataset == 'wikitext2':
            self.train_loader = get_dataloader(
                subset='train', 
                tokenizer=self.tokenizer, 
                batch_size=self.config.batch_size, 
                max_length=self.max_seq_len
            )
            self.valid_loader = get_dataloader(
                subset='valid', 
                tokenizer=self.tokenizer, 
                batch_size=self.config.batch_size, 
                max_length=self.max_seq_len
            )
        else:
            raise Exception()
    
    def get_optimizer(
        self,
        model:torch.nn.Module, 
        optimizer_type:str='AdamW',
        lr:float=1e-4,
        weight_decay:float=1e-3,
        no_decay_keywords=[]
    ):
        param_optimizer = list([(n, p) for n, p in model.named_parameters() if p.requires_grad])
        no_decay = [
            'bias', 
            'LayerNorm.bias', 
            'LayerNorm.weight', 
            'BatchNorm1d.weight', 
            'BatchNorm1d.bias', 
            'BatchNorm1d',
            'bnorm',
        ]
        high_lr = self.config.high_lr_names
        if no_decay_keywords is not None and len(no_decay_keywords) > 0:
            no_decay += no_decay_keywords
        set_normal = set([p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))])
        set_normal_no_wd = set([p for n, p in param_optimizer if any(nd in n for nd in no_decay)])
        set_high = set([p for n, p in param_optimizer if any(nk in n for nk in high_lr) and (not any(nd in n for nd in no_decay))])
        set_high_no_wd = set([p for n, p in param_optimizer if any(nk in n for nk in high_lr) and any(nd in n for nd in no_decay)])
        set_normal = set_normal - set_high
        set_normal_no_wd = set_normal_no_wd - set_high_no_wd
        params = [
            {'params': list(set_normal), 'weight_decay': weight_decay, 'lr': lr},
            {'params': list(set_normal_no_wd), 'weight_decay': 0.0, 'lr': lr},
            {'params': list(set_high), 'weight_decay': weight_decay, 'lr': lr*10},
            {'params': list(set_high_no_wd), 'weight_decay': 0.0, 'lr': lr*10},
        ]

        kwargs = {
            'lr':lr,
            'weight_decay':weight_decay,
        }
        
        if optimizer_type == 'AdamW':
            optim_cls = torch.optim.AdamW
        elif optimizer_type == 'Adam':
            optim_cls = torch.optim.Adam
        else: raise Exception()
        
        return optim_cls(params, **kwargs)
    
    def init_optimizer(self):
        self._istep = 0
        self.step = 0
        self.epoch = 0
        
        self.optimizer = self.get_optimizer(
            model=self.model, 
            lr=self.config.lr, 
            weight_decay=self.config.wd
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp_enabled)
        self.optimizer.zero_grad()
    
    def train_step(self, batch) -> Tuple[float, Dict[str, float]]:
        batch = batch_to(batch, self.device)
        del batch['trg_len']
        batch.update({
            'output_hidden_states': True,
            'output_attentions': True,
        })
        
        with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
            output_teacher = self.base_model(**batch)
        with torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
            batch['teacher'] = self.base_model
            output_student = self.model(**batch)
        
        if self.config.using_loss:
            if self.config.using_kd:
                loss_model = output_student.loss * 0.1
            else:
                loss_model = output_student.loss
        else:
            loss_model = 0.0
        
        loss_kd = 0
        if self.config.using_kd:
            for ilayer in range(len(output_teacher.hidden_states)):
                loss_kd += torch.nn.functional.mse_loss(output_teacher.hidden_states[ilayer], output_student.hidden_states[ilayer])
            loss_kd = loss_kd / len(output_teacher.hidden_states) * 10
            assert len(output_teacher.hidden_states) > 0
            loss_kd = loss_kd + torch.nn.functional.mse_loss(output_teacher.logits, output_student.logits) * 0.1
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            loss_special = self.model.calc_loss_special()
        
        loss = loss_model + loss_kd + loss_special
        
        self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
        
        if ((self._istep + 1) % self.config.gradient_accumulation_steps) == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            for module in self.model.modules():
                if hasattr(module, 'redraw_projections'):
                    module.redraw_projections(self.device)
        
        loss = loss.item()
        loss_details = {
            'loss': loss,
            'loss_sp': loss_special.item() if isinstance(loss_special, torch.Tensor) else loss_special, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
        }
        
        return loss, loss_details
    
    def train_epoch(self):
        m = Metric()
        
        train_loader_len = len(self.train_loader)
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar):
                wandb_dict = {}
                loss, loss_details = self.train_step(batch)
                wandb_dict['train/loss'] = loss
                wandb_dict['train/epoch'] = self.epoch + istep / train_loader_len
                wandb_dict.update({
                    f'trian/loss/{k}': v for k, v in loss_details.items()
                })
                
                if ((self.step + 1) % self.config.eval_steps) == 0:
                    score = self.evaluate()
                    wandb_dict['eval/score'] = score
                
                pbar.set_description(
                    f'[{self.epoch}/{self.config.epochs}] '\
                    f'L:{m.update(loss, "l"):.4f} '\
                    f'Lsp:{m.update(loss_details["loss_sp"], "sp"):.4f} '\
                    f'Lkd:{m.update(loss_details["loss_kd"], "kd"):.4f} '\
                    f'Lm:{m.update(loss_details["loss_model"], "md"):.4f}'
                )
                
                self._istep += 1
                self.step = self._istep // self.config.gradient_accumulation_steps
                
                if (self.step % self.config.wandb_steps) == 0 and (self._istep % self.config.gradient_accumulation_steps) == 0:
                    wandb.log(wandb_dict, step=self.step)
    
    def evaluate(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        self.base_model.eval()
        
        nlls = []
        for batch in tqdm.tqdm(self.valid_loader, dynamic_ncols=True, desc='evaluate_llm'):
            batch = batch_to(batch, self.device)
            trg_len = batch['trg_len']
            del batch['trg_len']
            batch.update({
                'output_hidden_states': True,
                'output_attentions': True,
            })
            with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
                self.base_model(**batch)
            with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
                batch['teacher'] = self.base_model
                output_student = self.model(**batch)
                neg_log_likelihood = output_student.loss * trg_len.item()
            nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).sum() / self.valid_loader.dataset.seq_len).item()
        print(f'[{self.epoch}/{self.config.epochs}] PPL:', ppl)
        return ppl
    
    def checkpoint_path(self):
        os.makedirs(f'./saves/trainer/opt_trainer/{self.config.experiment_name}/', exist_ok=True)
        path = f'./saves/trainer/opt_trainer/{self.config.experiment_name}/checkpoint.pth'
        return path
    
    def save(self, path=None):
        if path is None: path = self.checkpoint_path()
        torch.save({
            'step': self.step,
            '_istep': self._istep,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'base_model': self.base_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'model_config': self.config.model_config,
                'epochs': self.config.epochs,
            }
        }, path)
        print('saved', path)
    
    def load(self, path=None):
        if path is None: path = self.checkpoint_path()
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.scaler.load_state_dict(state['scaler'])
        self.optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        epoch = state['epoch']
        epochs = state['epochs']
        del state
        print(f'loaded {path}({step}@[{epoch}/{epochs}])')
    
    def main(self):
        from ..utils.secrets import WANDB_KEY, USER_NAME
        os.environ['WANDB_API_KEY'] = WANDB_KEY
        wandb.init(
            project=f"[{USER_NAME}] perlin-opt" if USER_NAME is not None else "perlin-opt",
            config=asdict(self.config)
        )
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.train_epoch()
            score = self.evaluate()
            wandb.log({'eval/score': score, 'train/epoch': self.epoch+1}, step=self.step)

if __name__ == '__main__':
    t = Trainer()
    t.main()