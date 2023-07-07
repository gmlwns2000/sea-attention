from dataset.test_batch_generator import load_test_batch
import torch
import matplotlib.pyplot as plt

# %matplotlib agg # NOTE check
def attentions_to_img(all_layers_attn_probs, dataset, subset, img_title):
    test_batch = load_test_batch(dataset, subset)
    stacked_attn_probs = torch.stack(all_layers_attn_probs, dim=1)
    # [batch_size, layer, head, length, length]
    batch_size = stacked_attn_probs.shape[0]
    layer_num = stacked_attn_probs.shape[1]
    head_num = stacked_attn_probs.shape[2]
    
    assert layer_num == len(all_layers_attn_probs)
    
    if batch_size == 1: # called in trainer
        batch_rows =1
        batch_cols =1
    else: # called in main
        assert batch_size == len(test_batch) == 10
        batch_rows, batch_cols = 2, 5
    assert batch_rows * batch_cols == batch_size

    rows, cols = layer_num, head_num
    plot = []
    batch_subtitle = []
    plot = []
    for b in range(batch_size): # TODO in trainer, max_length
        fig = plt.figure(figsize=(12,12), dpi=300)
        t = test_batch['attention_mask'][b]
        seq_len = (t==0).nonzero()[0].squeeze().item()
        batch_subtitle.append(f"test_batch_idx: {b}")
        i = 0
        for l in range(layer_num):
            for h in range(head_num):
                img = stacked_attn_probs[b,l,h,:seq_len,:seq_len]
                ax = fig.add_subplot(rows, cols, i+1)
                ax.set_title(f'l{l}_h{h+l*head_num}', fontsize=30)
                ax.imshow(img) # one atten_probs
                i+=1
        fig.tight_layout()
        plot.append(fig)

    batch_fig, axes = plt.subplots(batch_rows, batch_cols, figsize=(12*layer_num,12*head_num))# ,dpi=200 # layout="constrained"

    j = 0
    for r in range(batch_rows):
        for c in range(batch_cols):
            plot[j].canvas.draw()
            axes[r,c].imshow(plot[j].canvas.renderer.buffer_rgba()) # aspect="auto"
            axes[r,c].set_title(batch_subtitle[j], fontsize=50)
            axes[r,c].axis('off')
            axes[r,c].set_xticklabels([])
            axes[r,c].set_yticklabels([])
            j+=1
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,wspace=0, hspace=0) #  wspace=0, hspace=0
    plt.tight_layout() # batch_fig
    plt.suptitle(img_title, fontsize = 80)
    return plt