from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size
import torch
import os
from matplotlib import pyplot as plt
import wandb

from ..eda.viz_eda import register_event, dispatch

PERLIN_K_FLATTEN = True # token_wise if True
PERLIN_LAYERWISE = False
PERLIN_MODE = 'perlin'

bool2int = lambda x: 1 if x else 0

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        global PERLIN_LAYERWISE, PERLIN_MODE, PERLIN_K_FLATTEN

        task_to_batch_size['mnli'] = 16 if not PERLIN_LAYERWISE else 32

        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=False,
            trainer_name=f'perlin_trainer_kf{bool2int(PERLIN_K_FLATTEN)}_lw{bool2int(PERLIN_LAYERWISE)}_{PERLIN_MODE}',
            using_kd=(not PERLIN_LAYERWISE) and (PERLIN_MODE != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps=2000,
            lr = 1e-4,
            epochs = 20
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.perlin_mode = PERLIN_MODE
                module.perlin_k_flatten = PERLIN_K_FLATTEN
        
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
            self.model(**self.viz_batch) # (['labels'[16(batch_size)], 'input_ids'[16, 161], 'token_type_ids'[16, 161], 'attention_mask', 'output_hidden_states', 'output_attentions']
        # breakpoint()
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                if module.bert_attention_probs is not None:
                    bert_layer.append(module.bert_attention_probs) # [4(16), 12, 203, 203] = batch_size, head, length, length
                    # breakpoint()
                if  module.perlin_attention_probs is not None:
                    perlin_layer.append(module.perlin_attention_probs)  
                    # breakpoint()       
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
        if self.batch_size > 4:
            self.batch_for_viz = 5
        else:
            self.batch_for_viz = self.batch_size
        
        # bert
        for b in range(1): # self.batch_for_viz: using only some among self.bert_batch_size
            wandb_all_layers = []
            # self.batch_index = self.batch_size-b-1
            self.batch_index = 14 # mnli_includes long sequence - TODO change to dictionary
            self.bert_attention_mask_indx = self.viz_batch['attention_mask'][self.batch_index].shape[0] # inlcude padding
            for l in range(self.bert_layer_count):
                # breakpoint()
                bert_layerwise_matrix=[]
                for h in range(self.bert_head_count):
                    t1 = self.viz_batch['attention_mask'][self.batch_index]
                    self.bert_attention_mask_indx = (t1==0).nonzero()[0].squeeze().item()
                    
                    img = self.bert_layerwise_attention[self.batch_index, l, h, :self.bert_attention_mask_indx , :self.bert_attention_mask_indx]
                    # breakpoint()
                    img = img.detach().cpu().numpy()
                    # breakpoint()
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
                plt.suptitle(current_state+f":{self.batch_index+1}_{l+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                self.save_bert_path = f'./saves/trainer/bert_glue_trainer/bertM/{self.batch_index+1}_{l+1}.png'
                plt.savefig(self.save_bert_path, dpi=160)
                wandb_all_layers.append(wandb.Image(self.save_bert_path))    
            wandb.log({"bertM": wandb_all_layers})
        
        # perlin
        for b in range(1): # self.batch_for_viz: using only some among self.perlin_batch_size
            wandb_all_layers = []
            # self.batch_index = self.batch_size-b-1
            self.batch_index = 14 # mnli_includes long sequence - TODO change to dictionary TODO rename from difference with bert
            self.perlin_attention_mask_indx = self.viz_batch['attention_mask'][self.batch_index].shape[0] # inlcude padding
            for l in range(self.perlin_layer_count):
                # breakpoint()
                perlin_layerwise_matrix=[]
                for h in range(self.perlin_head_count):
                    t2 = self.viz_batch['attention_mask'][self.batch_index]
                    self.perlin_attention_mask_indx = (t2==0).nonzero()[0].squeeze().item()
                    
                    img = self.perlin_layerwise_attention[self.batch_index, l, h, :self.perlin_attention_mask_indx , :self.perlin_attention_mask_indx]
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
                plt.suptitle(current_state+f":{self.batch_index+1}_{l+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                self.save_perlin_path = './saves/trainer/bert_glue_trainer/perlinM/'
                if not PERLIN_K_FLATTEN: # column_wise
                    self.save_perlin_path += f'column_wise/{self.batch_index+1}_{l+1}.png'
                else: # token_wise
                    self.save_perlin_path += f'token_wise/{self.batch_index+1}_{l+1}.png'
                    
                plt.savefig(self.save_perlin_path, dpi=160)
                wandb_all_layers.append(wandb.Image(self.save_perlin_path))
            wandb.log({"perlinM": wandb_all_layers})
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='perlin', type=str)
    parser.add_argument('--layerwise', action='store_true', default=False)
    parser.add_argument('--k-colwise', action='store_true', default=False)
    args = parser.parse_args()
    
    PERLIN_MODE = args.mode
    PERLIN_K_FLATTEN = not args.k_colwise
    PERLIN_LAYERWISE = args.layerwise
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()