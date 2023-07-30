from ...utils import get_bench
from ..visualize.glue import main as visualize_main
from ..visualize.glue import add_perlin_model_options, parse_perlin_model_options
import argparse, os, torch
import matplotlib.pyplot as plt

bench = get_bench()
bench.activate_temp_buffers = True

parser = argparse.ArgumentParser()
    
parser.add_argument('--subset', type=str, default='mnli')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
add_perlin_model_options(parser)

args = parser.parse_args()
print(args)

kwargs = parse_perlin_model_options(args)
kwargs.update({
    'subset': args.subset,
    'checkpoint_path': args.checkpoint,
    'evaluate': args.evaluate
})

visualize_main(**kwargs)

index_layer = 7

attention_mask = bench.get_temp_buffer('attention_mask', index_layer)
# estimated : right after prediction
estimated_attention_score = bench.get_temp_buffer('estimated_attention_score', index_layer)
estimated_attention_probs = bench.get_temp_buffer('estimated_attention_probs', index_layer) # masked in 594
masked_estimated_attention_probs = bench.get_temp_buffer('masked_estimated_attention_probs', index_layer) # must be same with estimated_attention_probs, delete this

# estimated : after resized
estimated_attention_score_resized = bench.get_temp_buffer('estimated_attention_score_resized', index_layer)
estimated_attention_probs_resized = bench.get_temp_buffer('estimated_attention_probs_resized', index_layer)
# teacher : T*T, T*T_M
attention_probs_truth = bench.get_temp_buffer('attention_probs_truth', index_layer)
attention_probs_truth_m = bench.get_temp_buffer('attention_probs_truth_m', index_layer)
# "sum_mask" : col_select_mask
if kwargs['perlin_colsel_method'] == 'sum_mask':
    col_select_mask = bench.get_temp_buffer('col_select_mask', index_layer)
# large_inx_mask
large_inx_mask = bench.get_temp_buffer('large_inx_mask', index_layer)
# col_sel_estimated_attention_probs
col_sel_estimated_attention_probs_bef_select = bench.get_temp_buffer('col_sel_estimated_attention_probs_bef_select', index_layer)
col_sel_estimated_attention_probs_selcol_filled = bench.get_temp_buffer('col_sel_estimated_attention_probs_selcol_filled', index_layer)
# t_dead_mask : topk applied (partial_attention_mask)
t_dead_mask = bench.get_temp_buffer('t_dead_mask', index_layer)
# partial_attention_mask after topk, colsel (before interp)
partial_attention_mask_before_interp = bench.get_temp_buffer('partial_attention_mask_before_interp', index_layer)
# partial_attention_mask after topk, colsel (after interp)
partial_attention_mask = bench.get_temp_buffer('partial_attention_mask', index_layer)

# original dense probs
attention_probs_dense = bench.get_temp_buffer('attention_probs_dense', index_layer)
# partial_attention_probs
partial_attention_probs = bench.get_temp_buffer('attention_matrix', index_layer)

# partial_context_layer (not mandatory)
partial_context_layer = bench.get_temp_buffer('partial_context_layer', index_layer) # N, T, H*HID

# finally outputed estimated_attention_probs
estimated_attention_probs_for_output = bench.get_temp_buffer('estimated_attention_probs_for_output', index_layer)

N, T = attention_mask.shape[0], attention_mask.shape[-1]

def imsave(img: torch.Tensor, path):
    plt.clf()
    plt.imshow(img.cpu().numpy())
    plt.colorbar()
    plt.savefig(path, dpi=300)
    print(f'saved {path}')

root = './saves/tests/test_perlin_col_sel_batch/'
os.makedirs(root, exist_ok=True)

index_batch = 3
index_head = 2

imsave(estimated_attention_score[index_batch,index_head], os.path.join(root, 'est_score.png'))
imsave(estimated_attention_probs[index_batch,index_head], os.path.join(root, 'est_probs.png'))
imsave(masked_estimated_attention_probs[index_batch,index_head], os.path.join(root, 'masked_est_probs.png'))

imsave(estimated_attention_score_resized[index_batch,index_head], os.path.join(root, 'est_score_resized.png'))
imsave(estimated_attention_probs_resized[index_batch,index_head], os.path.join(root, 'est_probs_resized.png'))

imsave(attention_probs_truth[index_batch,index_head], os.path.join(root, 'probs_truth.png'))
imsave(attention_probs_truth_m[index_batch,index_head], os.path.join(root, 'probs_truth_m.png'))

if kwargs['perlin_colsel_method'] == 'sum_mask':
    imsave(col_select_mask[index_batch], os.path.join(root, 'col_select_mask.png'))

imsave(large_inx_mask[index_batch], os.path.join(root, 'large_inx_mask.png'))
imsave(col_sel_estimated_attention_probs_bef_select[index_batch], os.path.join(root, 'colsel_est_probs_bef_select.png'))
imsave(col_sel_estimated_attention_probs_selcol_filled[index_batch], os.path.join(root, 'colsel_est_probs_bef_select.png'))

imsave(t_dead_mask[index_batch], os.path.join(root, 't_dead_mask.png'))
imsave(partial_attention_mask_before_interp[index_batch,index_head], os.path.join(root, 'part_attn_mask_bef_interp.png'))
imsave(partial_attention_mask[index_batch,index_head], os.path.join(root, 'part_attn_mask_aft_interp.png'))

imsave(attention_probs_dense[index_batch,index_head], os.path.join(root, 'attn_dense.png'))
imsave(partial_attention_probs[index_batch,index_head], os.path.join(root, 'part_attn_probs.png'))
imsave(partial_context_layer[index_batch], os.path.join(root, 'part_context_layer.png'))

imsave(estimated_attention_probs_for_output[index_batch,index_head], os.path.join(root, 'est_output.png'))


# imsave(
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)) /\
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)).sum(dim=-1, keepdim=True), os.path.join(root, 'est_interp_norm.png'))
# imsave(partial_attention_mask_before_interp[index_batch,index_head], os.path.join(root, 'part.png'))
# imsave(partial_attention_mask[index_batch,index_head], os.path.join(root, 'part_interp.png'))
# imsave(attention_mask[index_batch,0].expand(T, T), os.path.join(root, 'attn_mask.png'))
# imsave(partial_attention_probs[index_batch,0].expand(T, T), os.path.join(root, 'final_partial_probs.png'))