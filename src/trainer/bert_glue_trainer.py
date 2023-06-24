import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import transformers
from datasets import load_dataset, load_metric
import random, copy
import torch

from transformers.models.bert import modeling_bert as berts
from ..models import perlin_bert as plberts
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
    breakpoint()
    if subset == 'bert':
        subset = "cola" #return dummy set
    breakpoint()
    
    dataset = load_dataset('glue', subset, split=split, cache_dir='./cache/datasets') # glue, mnli, train
    breakpoint()
    
    sentence1_key, sentence2_key = task_to_keys[subset]
    breakpoint()

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        ) # get examples containing 'premise', 'hypothesis', 'label', 'idx', 'labels'
        # breakpoint()
        result = tokenizer(*args, padding=True, max_length=256, truncation=True) # get result with 'input_ids', 'token_type', 'attention_mask'
        # breakpoint()
        # result = tokenizer(*args, padding="max_length", max_length=512, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    if split.startswith('train'): #shuffle when train set
        breakpoint()
        dataset = dataset.sort('label') # sort data according to column
        breakpoint()
        dataset = dataset.shuffle(seed=random.randint(0, 10000))
        breakpoint()
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, batch_size=128) # added labels as key
    breakpoint()
    dataset = dataset.map(encode, batched=True, batch_size=128) # iterates with each batch_size(128) # 여기 batch_size는 그냥 map할 때만 쓰는 거지 실제 batch_size(4)와 무관하며 의미없지?
    breakpoint() # how does dataset change?
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    breakpoint()

    dataloader = torch.utils.data.DataLoader(
        dataset, # features: ['premise', 'hypothesis', 'label', 'idx', 'labels', 'input_ids', 'token_type_ids', 'attention_mask'], num_rows: 392702
        batch_size=batch_size, # 4
        num_workers=0,
    )
    breakpoint()
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
        epochs = 100
    ) -> None:
        seed()
        breakpoint()
        
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
        
        breakpoint()
        self.reset_trainloader()
        self.valid_loader = get_dataloader(subset, self.tokenizer, self.batch_size, split=task_to_valid[self.subset])
        
        breakpoint()
        assert model_cls is not None
        self.model = model_cls(self.base_model.config)
        self.model.to(self.device)

        breakpoint()
        self.load_state_from_base()
        
        self.optimizer = self.get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.model_unwrap = self.model
        # self.model = torch.compile(self.model)
        breakpoint()
        
    def reset_trainloader(self):
        breakpoint()
        if self.subset != 'bert':
            self.train_loader = get_dataloader(self.subset, self.tokenizer, self.batch_size, split='train')
            breakpoint()
        else:
            self.train_loader = WikitextBatchLoader(self.batch_size)
            breakpoint()
    
    def load_state_from_base(self):
        load_result = self.model.load_state_dict(self.base_model.state_dict(), strict=False)
        for it in load_result.unexpected_keys:
            print('Trainer.init: unexpected', it)
        for it in load_result.missing_keys:
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
        # breakpoint()
        self.optimizer.zero_grad()
        # breakpoint()
        
        with torch.autocast('cuda', torch.float16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            output = self.model(**batch)
            with torch.no_grad():
                output_base = self.base_model(**batch)
        # breakpoint()
        
        if not self.subset == 'bert' and self.using_loss:
            loss_model = output.loss # output type(dictionary, list etc) consider해야 하지 않나???
        else:
            loss_model = 0.0
        # breakpoint()
        
        loss_kd = 0
        if self.using_kd:
            for ilayer in range(len(output_base.hidden_states)): # len 13 tuple, each element with [batch_size(2), 203, 768]
                loss_kd += torch.nn.functional.mse_loss(output_base.hidden_states[ilayer], output.hidden_states[ilayer]) # 각 [2, 203, 768]를 비교
            loss_kd = loss_kd / len(output_base.hidden_states) * 10
            assert len(output_base.hidden_states) > 0
        # breakpoint()
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            warnings.warn('special loss found!')
            loss_special = self.model.calc_loss_special()
        # breakpoint()
        
        loss = loss_model + loss_kd + loss_special
        # breakpoint()
        
        self.scaler.scale(loss).backward()
        # breakpoint()
        
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        # breakpoint()
        self.scaler.update()
        # breakpoint()
        
        self.loss = loss.item()
        # breakpoint()
        self.loss_details = {
            'loss': loss.item(), 
            'loss_sp': loss_special.item() if isinstance(loss_special, torch.Tensor) else loss_special, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
        }
        # breakpoint()
        
        # self.debug_plot_perlim()
    
    def train_epoch(self):
        # self.model = torch.compile(self.model_unwrap)
        breakpoint()
        self.reset_trainloader() # train_loader는 매 epoch 마다 change하는 거임? <- 매번 다르게 shuffle해야 하므로?
        breakpoint()
        
        self.model.train()
        self.base_model.eval()
        
        smooth_loss_sum = 0
        smooth_loss_count = 0
        
        m = Metric()
        breakpoint()
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar): # batch containing batch_size amounts of 'labels', 'imput_ids', 'token_type_ids', 'attention_mask"
                # breakpoint()
                batch = batch_to(batch, self.device) # with cuda
                # breakpoint()
                self.train_step(batch)
                # breakpoint()
                
                smooth_loss_sum += self.loss
                smooth_loss_count += 1
                # breakpoint()
                pbar.set_description(
                    f'[{self.epoch+1}/{self.epochs}] '
                    f'({self.running_type}) ' if self.running_type is not None else ''
                    f'L:{smooth_loss_sum/smooth_loss_count:.6f}({m.update(self.loss, "loss"):.4f}) '
                    f'Lsp:{m.update(self.loss_details["loss_sp"], "loss_sp"):.4f} '
                    f'Lkd:{m.update(self.loss_details["loss_kd"], "loss_kd"):.4f}'
                )
                # breakpoint()
                
                if ((istep+1) % self.eval_steps) == 0: # cross validation??
                    self.evaluate()
                    self.save()
                    self.model.train()
                    self.base_model.eval() # base_model은 계속 eval이었음에도 그냥 적어준 것?
                    m = Metric()
                # breakpoint()
            breakpoint() # *** 확인은 못 했으나 istep은 iteration 개수 즉 epoch / batch_size일듯
    
    def evaluate(self, max_step=123456789, show_messages=True, model=None, split='valid'):
        if self.subset == 'bert':
            return {'accuracy': 0.0}
        breakpoint()
        
        # seed()
        if model is None:
            model = self.model
        model.eval()
        breakpoint()
        
        if self.subset == 'bert':
            metric = load_metric('glue', 'cola')
        else:
            metric = load_metric('glue', self.subset)
        breakpoint()
        
        loader = self.valid_loader
        if split == 'train': #why???
            loader = self.train_loader
        breakpoint()
        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'({self.subset}[{split}])', dynamic_ncols=True)):
            if i > max_step: break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outputs = model(**batch)
            breakpoint()
            predictions = outputs[0]

            if self.subset != 'stsb': 
                predictions = torch.argmax(predictions, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        breakpoint()
        
        score = metric.compute()
        self.last_metric_score = score
        breakpoint()
        if show_messages:
            print('metric score', score)
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
    
    def debug_plot_perlim(self):
        os.makedirs('./saves/trainer/bert_glue_trainer/bertM/', exist_ok=True)
        os.makedirs('./saves/trainer/bert_glue_trainer/perlimM/', exist_ok=True)
        idx = 0
        idx_attn = 0
        layer_list = [] # check!
        for module in self.model.modules():
            if isinstance(module, plberts.BertSelfAttention):
                if module.bert_attention_probs is not None:
                    layer_list.append(module.bert_attention_probs)
        
        layerwise_attention = torch.stack(layer_list,dim=1) # batch_size, layer, head, length, length
        
        # bert : input same as        
        for i in range(self.batch_size): # check!~bert에서!!
            for j in range(12): # hard coded head size!
                for k in range(12): # hard coded head size!
                    img = layerwise_attention[i, j, k, : , :] # batch_size meaning???
                    img = img.detach().cpu().numpy()
                    
                    idx += 1
                    
                    plt.clf()
                    plt.imshow(img)
                    plt.colorbar()
                    plt.savefig(f'./saves/trainer/bert_glue_trainer/bertM/{i}_{j}_{k}_{idx}.png', dpi=160)
                        
                    if  module.perlin_attention_probs is not None:
                        for i in range(self.batch_size):
                            for j in range(12):           
                                img = module.perlin_attention_probs[i, j, :, :]
                                img = img.detach().cpu().numpy()
                                idx_attn += 1
                                
                                plt.clf()
                                plt.imshow(img)
                                plt.colorbar()
                                plt.savefig(f'./saves/trainer/bert_glue_trainer/perlimM/{i}_{j}_attn_{idx}.png', dpi=160)
    
    def main(self):
        self.epoch = 0
        self.step = 0
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            self.evaluate()
            # self.debug_plot_perlim()
            self.evaluate(split='train')
            self.save()

if __name__ == '__main__':
    breakpoint()
    trainer = Trainer(
        subset='mnli'
    )
    breakpoint()
    trainer.main()