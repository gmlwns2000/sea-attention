# 1. save png of every head per each layer's attention matrix
    # -> function of Trainer class
    # base model
    # perlin (columnwise / relationwise)
    # performer attention approximation
    
import os
import sys
import torch
from ..models import hf_bert as berts
from ..models import perlin_bert as perlin
import matplotlib.pyplot as plt
import wandb

# TODO modify base_model visualization? : visualize only once? or same as others?
# wandb for trainer
def plot_attentions_all_layer(self, current_state):
    '''
    input: trainer_name, attention_type
    output: attention_probs of every head per layer
    cases: base_mode / perlin (columnwise, relationwise) / performer
    python -m bert_glue_trainer for baseM
    python -m perlin_trainer for perlinM, performerM
    '''
    if 'bert_glue_trainer' in self.trainer_name: # base_model
        _ModelSelfAttention = berts.BertSelfAttention
        viz_matrix_type = "baseM"
    elif 'perlin_trainer' in self.trainer_name: # perlin or performer
        _ModelSelfAttention = perlin.BertSelfAttention
        if self.perlin_mode == 'perlin':
            if self.perlin_before_topk:
                viz_matrix_type = 'perlinM_before_topK'
            else:
                viz_matrix_type = 'perlinM_after_topK'
        elif self.perlin_mode == 'performer':
            viz_matrix_type = 'performerM'
        else:
            raise Exception(f"_Plot_trainer] self.perlin_mode {self.perlin_mode}")
    else: #TODO
        raise Exception(f"_Plot_trainer] {self.trainer_name} is neither bert_glue_trainer nor perlin_trainer.")
    
    os.makedirs(f'./saves/trainer/{self.trainer_name}/{viz_matrix_type}/{current_state}/', exist_ok=True)
    
    all_layers_attention_probs = []
    
    self.viz_batch['output_hidden_states'] = True
    self.viz_batch['output_attentions'] = True
    
    if 'bert_glue_trainer' not in self.trainer_name:
        self.viz_batch['teacher'] = self.base_model
        
    with torch.no_grad():
        self.model(**self.viz_batch) # (['labels'[16(batch_size)], 'input_ids'[16, 161], 'token_type_ids'[16, 161], 'attention_mask', 'output_hidden_states', 'output_attentions']
    
    for module in self.model.modules(): # base_model, perlin(), performer
        if isinstance(module, _ModelSelfAttention): # self._BertselfAttention
            if module.perlin_last_attention_prob is not None:
                all_layers_attention_probs.append(module.perlin_last_attention_prob) # [4(16), 12, 203, 203] = batch_size, head, length, length
    
    stack_all_layers_attention_probs = torch.stack(all_layers_attention_probs,dim=1) # batch_size, layer, head, length, length
    _batch_size = stack_all_layers_attention_probs.shape[0]
    _layer_num = stack_all_layers_attention_probs.shape[1]
    _head_num = stack_all_layers_attention_probs.shape[2]
    assert _layer_num == len(all_layers_attention_probs)
    assert _head_num % 2 == 0 # TODO check ~ generalization
    
    # when you want to viz with _batch_size_for_viz > 1
    # if _batch_size > 4:
    #     _batch_size_for_viz = 5
    # else:
    #     _batch_size_for_viz = _batch_size
    _batch_size_for_viz = 1
    
    for b in range(_batch_size_for_viz):
        wandb_all_layers_attention_probs = []
        
        if _batch_size_for_viz > 1:
            _batch_idx = b # _batch_size-b-1
        else:
            _batch_idx = 14 # TODO : change to dictionary : this is idx of long sequenece for mnli
            
        _attention_mask_idx = self.viz_batch['attention_mask'][_batch_idx].shape[0] # inlcude padding
        for l in range(_layer_num):
            _all_layers_attention_matrix=[]
            for h in range(_head_num):
                t1 = self.viz_batch['attention_mask'][_batch_idx]
                _attention_mask_idx = (t1==0).nonzero()[0].squeeze().item() # exclude padding
                
                img = stack_all_layers_attention_probs[_batch_idx, l, h, :_attention_mask_idx, :_attention_mask_idx]
                img = img.detach().cpu().numpy()
                
                plt.clf()
                _all_layers_attention_matrix.append(img)
            nrows=2
            ncols=_head_num//2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) # nrows*ncols==self.head_count
            for i, ax in enumerate(axes.flat):
                ax.imshow(_all_layers_attention_matrix[i])
                # plt.imshow(img)
                # plt.colorbar()
            for i in range(nrows):
                for j in range(ncols):
                    axes[i,j].set_title(f"Head:{ncols*i+j+1}")
            plt.suptitle(current_state+f":{_batch_idx+1}_{l+1}", fontsize=16)
            plt.tight_layout()
            plt.show()
            _saved_path = f'./saves/trainer/{self.trainer_name}/{viz_matrix_type}/{current_state}/{_batch_idx+1}_{l+1}.png'
            plt.savefig(_saved_path, dpi=160)
            wandb_all_layers_attention_probs.append(wandb.Image(_saved_path))
        wandb.log({viz_matrix_type: wandb_all_layers_attention_probs})

# use saved model (resuming starting from that trained point)
# TODO + use optimizer state_dict in bert_glue_trainer and update save() and loade() etc
