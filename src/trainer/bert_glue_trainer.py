import os
from typing import Callable, Generic, TypeVar
import warnings
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
import tqdm
import transformers
from datasets import load_dataset, load_metric
import random, copy
import torch
import wandb
from ..eda.viz_eda import dispatch

# from transformers.models.bert import modeling_bert as berts
from ..models import hf_bert as berts
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to, seed
from ..dataset.wikitext import WikitextBatchLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_epochs = {
    "cola": 100,
    "mnli": 40,
    "mrpc": 200,
    "qnli": 30,
    "qqp":  40,
    "rte":  200,
    "sst2": 150,
    "stsb": 200,
    "wnli": 200,
    "bert": 200,
}

task_to_batch_size = {
    "cola": 64,
    "mnli": 4,
    "mrpc": 32,
    "qnli": 4,
    "qqp":  16,
    "rte":  8,
    "sst2": 16,
    "stsb": 16,
    "wnli": 32,
    "bert": 4,
}

task_to_valid = {
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

def get_dataloader(subset, tokenizer, batch_size, split='train'):
    if subset == 'bert':
        subset = "cola" #return dummy set
    
    dataset = load_dataset('glue', subset, split=split, cache_dir='./cache/datasets')
    
    sentence1_key, sentence2_key = task_to_keys[subset]

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, max_length=256, truncation=True)
        # result = tokenizer(*args, padding="max_length", max_length=512, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    if split.startswith('train'): #shuffle when train set
        dataset = dataset.sort('label')
        dataset = dataset.shuffle(seed=random.randint(0, 10000))
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, batch_size=384)
    dataset = dataset.map(encode, batched=True, batch_size=384)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0,
    )
    return dataloader

def get_base_model(dataset, only_tokenizer=False):
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

    # NOTE(HJ): this bert models has special hooks
    model = {
        "cola": berts.BertForSequenceClassification,
        "mnli": berts.BertForSequenceClassification,
        "mrpc": berts.BertForSequenceClassification,
        "qnli": berts.BertForSequenceClassification,
        "qqp": berts.BertForSequenceClassification,
        "rte": berts.BertForSequenceClassification,
        "sst2": berts.BertForSequenceClassification,
        "stsb": berts.BertForSequenceClassification,
        "wnli": berts.BertForSequenceClassification,
        "bert": berts.BertForSequenceClassification,
    }[dataset]
    
    tokenizer = transformers.BertTokenizerFast.from_pretrained(checkpoint)
    if only_tokenizer:
        return None, tokenizer
    
    bert = model.from_pretrained(checkpoint, cache_dir='./cache/huggingface/')
    return bert, tokenizer

class Metric:
    def __init__(self):
        self.sum = {}
        self.count = {}
        
    def update(self, x, name='', weight=1):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if not name in self.sum:
            self.sum[name] = 0
            self.count[name] = 0
        self.sum[name] += x * weight
        self.count[name] += weight
        return self.sum[name] / self.count[name]

    def get(self, name=''):
        return self.sum[name] / self.count[name]

    def to_dict(self):
        r = {}
        for key in self.sum:
            r[key] = self.get(key)
        return r

class Trainer:
    def __init__(
        self, 
        subset='mrpc',
        model_cls=berts.BertForSequenceClassification,
        high_lr_names=[],
        amp_enabled = True,
        trainer_name = 'bert_glue_trainer',
        running_type = None,
        using_kd = True,
        using_loss = True,
        eval_steps = 1500,
        lr = 1e-5,
        epochs = 100,
        load_ignore_keys = ['perlin', 'pbert', 'permute']
    ) -> None:
        seed()
        
        self.load_ignore_keys = load_ignore_keys
        self.running_type = running_type
        self.trainer_name = trainer_name
        self.subset = subset
        self.high_lr_names = high_lr_names
        self.using_kd = using_kd
        self.using_loss = using_loss
        
        self.amp_enabled = amp_enabled
        self.device = 0
        
        self.batch_size = task_to_batch_size[self.subset]
        
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.lr = lr
        self.wd = 1e-2
        
        self.base_model, self.tokenizer = get_base_model(subset)
        self.base_model.to(self.device)
        
        self.reset_trainloader()
        self.valid_loader = get_dataloader(subset, self.tokenizer, self.batch_size, split=task_to_valid[self.subset])
        # breakpoint()
        
        assert model_cls is not None
        self.model = model_cls(self.base_model.config)
        self.model.to(self.device)

        self.load_state_from_base()
        
        self.optimizer = self.get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.model_unwrap = self.model
        # self.model = torch.compile(self.model)
        
        for batch in self.valid_loader:
            batch = batch_to(batch, self.device)
            self.viz_batch = batch
            break
        
        self.input_data = self.valid_loader.dataset
    
    def reset_trainloader(self):
        if self.subset != 'bert':
            self.train_loader = get_dataloader(self.subset, self.tokenizer, self.batch_size, split='train')
        else:
            self.train_loader = WikitextBatchLoader(self.batch_size)
    
    def load_state_from_base(self):
        load_result = self.model.load_state_dict(self.base_model.state_dict(), strict=False)
        for it in load_result.unexpected_keys:
            print('Trainer.init: unexpected', it)
        for it in load_result.missing_keys:
            if not any([k in it for k in self.load_ignore_keys]):
                print('Trainer.init: missing', it)
    
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
        high_lr = self.high_lr_names
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
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        with torch.autocast('cuda', torch.bfloat16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            with torch.no_grad():
                output_base = self.base_model(**batch)
            batch['teacher'] = self.base_model
            output = self.model(**batch)
        
        
        if not self.subset == 'bert' and self.using_loss:
            loss_model = output.loss
        else:
            loss_model = 0.0
        
        loss_kd = 0
        if self.using_kd:
            for ilayer in range(len(output_base.hidden_states)):
                loss_kd += torch.nn.functional.mse_loss(output_base.hidden_states[ilayer], output.hidden_states[ilayer])
            loss_kd = loss_kd / len(output_base.hidden_states) * 10
            assert len(output_base.hidden_states) > 0
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            # warnings.warn('special loss found!')
            loss_special = self.model.calc_loss_special()
            
        loss = loss_model + loss_kd + loss_special
        
        self.scaler.scale(loss).backward()
        
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.loss = loss.item()
        self.loss_details = {
            'loss': loss.item(), 
            'loss_sp': loss_special.item() if isinstance(loss_special, torch.Tensor) else loss_special, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
        }
    
    def train_epoch(self):
        # self.model = torch.compile(self.model_unwrap)
        self.reset_trainloader()
        
        self.model.train()
        self.base_model.eval()
        
        smooth_loss_sum = 0
        smooth_loss_count = 0
        
        m = Metric()
        
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                self.train_step(batch)
                
                smooth_loss_sum += self.loss
                smooth_loss_count += 1
                pbar.set_description(
                    f'[{self.epoch+1}/{self.epochs}] '
                    f'({self.running_type}) ' if self.running_type is not None else ''
                    f'L:{smooth_loss_sum/smooth_loss_count:.6f}({m.update(self.loss, "loss"):.4f}) '
                    f'Lsp:{m.update(self.loss_details["loss_sp"], "loss_sp"):.4f} '
                    f'Lkd:{m.update(self.loss_details["loss_kd"], "loss_kd"):.4f}'
                )
                # breakpoint()
                # dispatch('plot_perlin_attention_called', current_state=f"Testing!!!_{self.epoch+1}_{istep+1}")
                
                if ((istep+1) % self.eval_steps) == 0:
                    self.evaluate()
                    dispatch('plot_perlin_attention_called', current_state=f"train_epoch_{self.epoch+1}_{istep+1}")
                    # TODO plot train graph
                    
                    # self.plot_perlin_attention(current_state=f"train_epoch_{self.epoch+1}_{istep+1}") # call in model.eval() mode
                    self.save()
                    self.model.train()
                    self.base_model.eval()
                    m = Metric()
    
    def evaluate(self, max_step=123456789, show_messages=True, model=None, split='valid'):
        if self.subset == 'bert':
            return {'accuracy': 0.0}
        
        # seed()
        if model is None:
            model = self.model
        model.eval()
        
        if self.subset == 'bert':
            metric = load_metric('glue', 'cola')
        else:
            metric = load_metric('glue', self.subset)
        
        loader = self.valid_loader
        if split == 'train':
            loader = self.train_loader
        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'({self.subset}[{split}])', dynamic_ncols=True)):
            if i > max_step: break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']
            
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=self.amp_enabled):
                self.base_model(**batch)
                batch['teacher'] = self.base_model
                outputs = model(**batch)
            predictions = outputs[0]

            if self.subset != 'stsb': 
                predictions = torch.argmax(predictions, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        
        score = metric.compute()
        self.last_metric_score = score
        if show_messages:
            tqdm.tqdm.write(f'metric score {score}')
        return score

    def save(self):
        os.makedirs(f'./saves/trainer/{self.trainer_name}/', exist_ok=True)
        path = f'./saves/trainer/{self.trainer_name}/checkpoint_{self.subset}.pth'
        print(f'Trainer: save {path}')
        torch.save({
            'model': self.model.state_dict(),
            'base_model': self.base_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path=None):
        try:
            if path is None:
                path = f'./saves/trainer/{self.trainer_name}/checkpoint_{self.subset}.pth'
            print(f'Trainer: load {path}')
            state = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state['model'])
            self.base_model.load_state_dict(state['base_model'])
            # self.optimizer.load_state_dict(state['optimizer'])
            del state
        except Exception as ex:
            print('error while load', ex)
    
    def plot_base_attention(self, current_state):
        os.makedirs('./saves/trainer/bert_glue_trainer/base_model/', exist_ok=True)
        
        base_layer = []
        
        with torch.no_grad():
            output_base = self.base_model(**self.viz_batch) # why not using output base..?
        
        for module in self.base_model.modules():
            if isinstance(module, berts.BertSelfAttention):
                if self.perlin_last_attention_prob is not None:
                    base_layer.append(module.perlin_last_attention_prob)
        
        self.base_layerwise_attention = torch.stack(base_layer,dim=1) # batch_size, layer, head, length, length
        self.base_batch_size = self.base_layerwise_attention.shape[0]
        self.base_layer_count = self.base_layerwise_attention.shape[1]
        self.base_head_count = self.base_layerwise_attention.shape[2]
        assert self.base_layer_count == len(base_layer)
        assert self.base_head_count % 2 == 0 # TODO check! ~ generalization
        
        for b in range(1): # self.batch_for_viz: using only some among self.bert_batch_size
            wandb_all_layers = []
            # self.batch_index = self.batch_size-b-1
            self.batch_index = 14 # mnli_includes long sequence - TODO change to dictionary
            self.base_attention_mask_indx = self.viz_batch['attention_mask'][self.batch_index].shape[0] # inlcude padding
            for l in range(self.base_layer_count):
                # breakpoint()
                base_layerwise_matrix=[]
                for h in range(self.base_head_count):
                    t1 = self.viz_batch['attention_mask'][self.batch_index]
                    self.base_attention_mask_indx = (t1==0).nonzero()[0].squeeze().item()
                    
                    img = self.base_layerwise_attention[self.batch_index, l, h, :self.base_attention_mask_indx , :self.base_attention_mask_indx]
                    # breakpoint()
                    img = img.detach().cpu().numpy()
                    # breakpoint()
                    idx += 1
                    
                    plt.clf()
                    base_layerwise_matrix.append(img)
                nrows=2
                ncols=self.base_head_count//2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) # nrows*ncols==self.head_count
                for i, ax in enumerate(axes.flat):
                    ax.imshow(base_layerwise_matrix[i])
                    # plt.imshow(img)
                    # plt.colorbar()
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i,j].set_title(f"Head:{ncols*i+j+1}")
                plt.suptitle(current_state+f":{self.batch_index+1}_{l+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                self.save_base_path = f'./saves/trainer/bert_glue_trainer/baseM/{self.batch_index+1}_{l+1}.png'
                plt.savefig(self.save_base_path, dpi=160)
                wandb_all_layers.append(wandb.Image(self.save_base_path))    
            wandb.log({"baseM": wandb_all_layers})
    
    
    def main(self):
        # wandb.login()
        # wandb.init( # TODO change
        #     project="[baseM] visualize_bert_perlin"
        # )
        # plot_base_model
        # self.plot_base_attention(current_state="baseM_main")
        # print("All done!!")
        
        self.epoch = 0
        self.step = 0
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            self.evaluate()
            dispatch('plot_perlin_attention_called', current_state=f"main_epoch_{self.epoch+1}")
            # self.plot_perlin_attention(current_state=f"main_epoch_{self.epoch+1}")
            self.evaluate(split='train') # for checking overfitting
            self.save()

# TODO
'''
T = TypeVar("T")
class EventHandler(Generic[T]):
    def __init__(self) -> None:
        self.cbs = []
    
    def invoke(self, *args, **kwargs):
        for cb in self.cbs:
            cb(*args, **kwargs)
    
    def add(self, cb: Callable[[T], None]):
        self.cbs.append(cb)

@dataclass
class TrainerCallbacks:
    train_finished = EventHandler[float]()
    epoch_finished = EventHandler[float]()
    on_evaluate = EventHandler[float]()
    
# outside
callback = TrainerCallbacks()
callback.on_evaluate.add(lambda msg, loss, acc: print(msg))
# Trainer(callback=callback)

# def evaluate() ...
callback.on_evaluate.invoke("message", loss=0.1, accuracy=0.9)
'''

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()