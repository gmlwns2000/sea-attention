from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size
import torch
import os
from matplotlib import pyplot as plt
import wandb

from ..eda.viz_eda import register_event, dispatch

PERLIN_LAYERWISE = False
PERLIN_MODE = 'perlin'

task_to_batch_size['mnli'] = 16 if not PERLIN_LAYERWISE else 32

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=False,
            trainer_name='perlin_trainer',
            using_kd=(not PERLIN_LAYERWISE) and (PERLIN_MODE != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps=2000,
            lr = 1e-4,
            epochs = 20
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.perlin_mode = PERLIN_MODE
        
        if PERLIN_LAYERWISE:
            for module in self.model.modules():
                if isinstance(module, perlin.BertSelfAttention):
                    module.perlin_layerwise = True
            
            for name, param in self.model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # eda
        register_event('plot_perlin_attention_called',self.plot_perlin_attention)
        
                    
    def plot_perlin_attention(self, current_state):
        os.makedirs('./saves/trainer/bert_glue_trainer/bertM/', exist_ok=True)
        os.makedirs('./saves/trainer/bert_glue_trainer/perlimM/', exist_ok=True)
        idx = 0
        idx_attn = 0
        bert_layer = []
        perlin_layer = []
        
        self.viz_batch['output_hidden_states'] = True
        self.viz_batch['output_attentions'] = True
            
        with torch.no_grad():
            self.model(**self.viz_batch)
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                if module.bert_attention_probs is not None:
                    bert_layer.append(module.bert_attention_probs) # [4(16), 12, 203, 203] = batch_size, head, length, length
                if  module.perlin_attention_probs is not None:
                    perlin_layer.append(module.perlin_attention_probs)         
        # len(bert_list) == 12 : seems it contains all layer for forward
        
        self.bert_layerwise_attention = torch.stack(bert_layer,dim=1) # batch_size, layer, head, length, length
        self.bert_batch_size = self.bert_layerwise_attention.shape[0]
        self.bert_layer_count = self.bert_layerwise_attention.shape[1]
        self.bert_head_count = self.bert_layerwise_attention.shape[2]
        assert self.bert_layer_count == len(bert_layer)
        assert self.bert_head_count % 2 == 0 # TODO check! ~ generalization
        # breakpoint()
        
        self.perlin_layerwise_attention = torch.stack(perlin_layer,dim=1) # batch_size, layer, head, length, length
        # breakpoint()
        self.perlin_batch_size = self.perlin_layerwise_attention.shape[0]
        self.perlin_layer_count = self.perlin_layerwise_attention.shape[1]
        self.perlin_head_count = self.perlin_layerwise_attention.shape[2]
        assert self.perlin_layer_count == len(perlin_layer)
        assert self.perlin_head_count % 2 == 0 # TODO check! ~ generalization
        
        # bert
        for b in range(self.bert_batch_size):
            for l in range(self.bert_layer_count):
                # breakpoint()
                bert_layerwise_matrix=[]
                for h in range(self.bert_head_count):
                    img = self.bert_layerwise_attention[b, l, h, : , :] # batch_size 2나 4 인거 맞음...?
                    # breakpoint()
                    img = img.detach().cpu().numpy()
                    
                    idx += 1
                    
                    plt.clf()
                    bert_layerwise_matrix.append(img)
                nrows=2
                ncols=self.bert_head_count//2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) # nrows*ncols==self.head_count
                for i, ax in enumerate(axes.flat):
                    ax.imshow(bert_layerwise_matrix[i])
                    # plt.imshow(img)
                    # plt.colorbar()
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i,j].set_title(f"Head:{ncols*i+j+1}")
                plt.suptitle(current_state+f":{b+1}_{l+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                plt.savefig(f'./saves/trainer/bert_glue_trainer/bertM/{b+1}_{l+1}.png', dpi=160)
                wandb.log({"bertM": [wandb.Image(f'./saves/trainer/bert_glue_trainer/bertM/{b+1}_{l+1}.png')]})
        
        # perlin
        for b in range(self.perlin_batch_size):
            for l in range(self.perlin_layer_count):
                # breakpoint()
                perlin_layerwise_matrix=[]
                for h in range(self.perlin_head_count):
                    img = self.perlin_layerwise_attention[b, l, h, : , :] #batch_size 2나 4 인거 맞음...?
                    # breakpoint()
                    img = img.detach().cpu().numpy()
                    
                    idx += 1
                    
                    plt.clf()
                    perlin_layerwise_matrix.append(img)
                nrows=2
                ncols=self.perlin_head_count//2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) # nrows*ncols==self.head_count
                for i, ax in enumerate(axes.flat):
                    ax.imshow(perlin_layerwise_matrix[i])
                    # plt.imshow(img)
                    # plt.colorbar()
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i,j].set_title(f"Head:{ncols*i+j+1}")
                plt.suptitle(current_state+f":{b+1}_{l+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                plt.savefig(f'./saves/trainer/bert_glue_trainer/perlinM/{b+1}_{l+1}.png', dpi=160)
                wandb.log({"perlinM": [wandb.Image(f'./saves/trainer/bert_glue_trainer/perlinM/{b+1}_{l+1}.png')]})
    

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()