import os
import warnings
from matplotlib import pyplot as plt
from ..dataset.test_batch_save_load import load_test_batch
from ..main.evaluation import get_attns_img
import numpy as np
import tqdm
import transformers
from datasets import load_dataset, load_metric
import random, copy
import torch
import wandb

# from transformers.models.bert import modeling_bert as berts
from ..models import hf_bert as berts
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to, seed
from ..dataset.wikitext import WikitextBatchLoader

bool2int = lambda x: 1 if x else 0

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
    "mnli": 16, # NOTE(JIN): changed 4 to 16 TODO check
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

BASE_MODEL_TYPE = 'bert'
DATASET = 'glue'
TEST_BATCH_SIZE = 1
assert TEST_BATCH_SIZE == 1 # attentions_to_img.py
FOR_EVAL = False
assert not FOR_EVAL

def get_dataloader(subset, tokenizer, batch_size, split='train', making_test_batch=False):
    if subset == 'bert':
        subset = "cola" #return dummy set
    
    dataset = load_dataset(DATASET, subset, split=split, cache_dir='./cache/datasets')
    
    sentence1_key, sentence2_key = task_to_keys[subset]

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        if not making_test_batch:
            result = tokenizer(*args, padding=True, max_length=256, truncation=True)
        else:
            result = tokenizer(*args, padding='max_length', max_length=256, truncation=True)
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
        num_workers=0, # NOTE(JIN): check value
    )
    return dataloader

def get_base_model(subset, only_tokenizer=False):
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
    }[subset]

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
    }[subset]
    
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
        load_ignore_keys = ['perlin', 'pbert', 'permute'],
        attention_method = 'perlin',
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

        self.attention_method = attention_method
        
        self.base_model, self.tokenizer = get_base_model(subset)
        self.base_model.to(self.device)
        
        self.reset_trainloader()
        self.valid_loader = get_dataloader(subset, self.tokenizer, self.batch_size, split=task_to_valid[self.subset])
        
        test_batch = load_test_batch(DATASET, self.subset, FOR_EVAL, TEST_BATCH_SIZE)
        
        print("\n\nBert_glue_trainer] test_batch", test_batch) # for debug

        test_batch = batch_to(test_batch, self.device) # TODO check <- after saving

        assert model_cls is not None
        self.model = model_cls(self.base_model.config)
        self.model.to(self.device)

        self.load_state_from_base()
        
        self.optimizer = self.get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.model_unwrap = self.model
        # self.model = torch.compile(self.model)
    
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
            for ilayer in range(len(output_base.hidden_states)): # NOTE(JIN): first is output of BertEmbeddings. len: 13 tensor, each [batch_size, sequence_lengh, hidden_size]
                loss_kd += torch.nn.functional.mse_loss(output_base.hidden_states[ilayer], output.hidden_states[ilayer])
            loss_kd = loss_kd / len(output_base.hidden_states) * 10
            assert len(output_base.hidden_states) > 0
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            warnings.warn('special loss found!')
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
                
                if ((istep+1) % self.eval_steps) == 0:
                    self.evaluate()
                    img_title = f"train_epoch/ep{self.epoch}_st{self.step}_lr{self.lr}"

                    dense_attns_img, sparse_attns_img = get_attns_img(
                        BASE_MODEL_TYPE,
                        DATASET, 
                        self.subset, 
                        self.attention_method, # it wouldn't be "base"
                        self.model, 
                        self.base_model,
                        img_title,
                        TEST_BATCH_SIZE,
                        FOR_EVAL)

                    if dense_attns_img is not None:
                        wandb.log({self.attention_method : dense_attns_img})
                    if sparse_attns_img is not None:
                        wandb.log({self.attention_method : sparse_attns_img})

                    self.save()
                    
                    self.model.train()
                    self.base_model.eval()
                    m = Metric()
                    
                    # visualization
    
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
            
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=self.amp_enabled): # TODO JIN: modified torch.bfloat16
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

    def save(self): # TODO(JIN): why not save the entire model?
        os.makedirs(f'./saves/trainer/{self.trainer_name}/{DATASET}/', exist_ok=True)
        path = f'./saves/trainer/{self.trainer_name}/{DATASET}/checkpoint_{self.subset}.pth' # NOTE(JIN): update plot.py once changed
        print(f'Trainer: save {path}')
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'lr': self.lr,
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

    def main(self):
        run = wandb.init(
             project="perlin-glue",
             config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "subset": self.subset,
                "epochs": self.epochs,
            }
         )
        wandb.watch(self.model, log='all')
    
        self.epoch = 0
        self.step = 0
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            self.evaluate()

            img_title = f"main_epoch/ep{self.epoch}_st{self.step}_lr{self.lr}"
            dense_attns_img, sparse_attns_img = get_attns_img(
                BASE_MODEL_TYPE,
                DATASET, 
                self.subset, 
                self.attention_method, # it wouldn't be "base"
                self.model, 
                self.base_model, 
                img_title,
                TEST_BATCH_SIZE,
                FOR_EVAL)
            
            if dense_attns_img is not None:
                wandb.log({self.attention_method : dense_attns_img})
            if sparse_attns_img is not None:
                wandb.log({self.attention_method : sparse_attns_img})
            
            self.evaluate(split='train') # check overfitting
            self.save()

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()