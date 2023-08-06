from ....utils import get_bench
from ...visualize.glue import main as visualize_main
from ...visualize.glue import add_perlin_model_options, parse_perlin_model_options
import argparse, os, torch
import matplotlib.pyplot as plt
from ...visualize.common import convert_to_colormap
import cv2

# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()
#     torch.multiprocessing.set_start_method('spawn') #, force=True
    

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
estimated_attention_score_init = bench.get_temp_buffer('estimated_attention_score_init', index_layer)
estimated_attention_probs_init = bench.get_temp_buffer('estimated_attention_probs_init', index_layer) # masked in 594

# teacher : T*T, T*T_M
attention_scores_truth = bench.get_temp_buffer('attention_scores_truth', index_layer)
attention_probs_truth = bench.get_temp_buffer('attention_probs_truth', index_layer)
attention_probs_truth_m = bench.get_temp_buffer('attention_probs_truth_m', index_layer)

# estimated : after resized
estimated_attention_score_resized = bench.get_temp_buffer('estimated_attention_score_resized', index_layer)
estimated_attention_probs_resized = bench.get_temp_buffer('estimated_attention_probs_resized', index_layer)

masked_estimated_attention_probs = bench.get_temp_buffer('masked_estimated_attention_probs', index_layer) # must be same with estimated_attention_probs, delete this

if kwargs['perlin_colsel']:
    per_item_col_thickness = bench.get_temp_buffer('per_item_col_thickness', index_layer)
    col_mean_in_flatten = bench.get_temp_buffer('col_mean_in_flatten', index_layer)
    col_result_mask = bench.get_temp_buffer('col_result_mask', index_layer)
    col_result_mask_cnt = bench.get_temp_buffer('col_result_mask_cnt', index_layer)
    inflated_per_item_col = bench.get_temp_buffer('inflated_per_item_col', index_layer)
    print('per_item_col_thickness', per_item_col_thickness)
    print('col_mean_in_flatten', col_mean_in_flatten)
    print('col_result_mask', col_result_mask)
    print('col_result_mask_cnt', col_result_mask_cnt)
    print('inflated_per_item_col', inflated_per_item_col)

    colsel_perform_cnt = bench.get_temp_buffer('colsel_perform_cnt', index_layer)
    per_item_col_real_1 = bench.get_temp_buffer('per_item_col_real_1', index_layer)
    col_t_alive_mask_bef_inter_1 = bench.get_temp_buffer('col_t_alive_mask_bef_inter_1', index_layer)
    col_t_alive_mask_t_1 = bench.get_temp_buffer('col_t_alive_mask_t_1', index_layer)
    col_t_alive_mask_m_1 = bench.get_temp_buffer('col_t_alive_mask_m_1', index_layer)
    per_item_condition_not_satisfied_1 = bench.get_temp_buffer('per_item_condition_not_satisfied_1', index_layer)
    print('colsel_perform_cnt', colsel_perform_cnt)
    print('per_item_col_real_1', per_item_col_real_1)
    print('per_item_condition_not_satisfied_1', per_item_condition_not_satisfied_1)

    if colsel_perform_cnt>0:
        layer_id_list = bench.get_temp_buffer('layer_id')
        colsel2_i = layer_id_list.index(index_layer)
        print('layer_id_list', layer_id_list)
        print('colsel2_i', colsel2_i)
        per_item_col_real_1_for2 = bench.get_temp_buffer('per_item_col_real_1_for2', colsel2_i)
        col_t_alive_mask_bef_inter_1_for2 = bench.get_temp_buffer('col_t_alive_mask_bef_inter_1_for2', colsel2_i)
        col_t_alive_mask_t_1_for2 = bench.get_temp_buffer('col_t_alive_mask_t_1_for2', colsel2_i)
        col_t_alive_mask_m_1_for2 = bench.get_temp_buffer('col_t_alive_mask_m_1_for2', colsel2_i)
        per_item_condition_not_satisfied_1_for2 = bench.get_temp_buffer('per_item_condition_not_satisfied_1_for2', colsel2_i)
        print('per_item_col_real_1_for2', per_item_col_real_1_for2)
        print('per_item_condition_not_satisfied_1_for2', per_item_condition_not_satisfied_1_for2)

        per_item_col_real_2 = bench.get_temp_buffer('per_item_col_real_2', colsel2_i)
        col_t_alive_mask_bef_inter_2 = bench.get_temp_buffer('col_t_alive_mask_bef_inter_2', colsel2_i)
        col_t_alive_mask_t_2 = bench.get_temp_buffer('col_t_alive_mask_t_2', colsel2_i)
        col_t_alive_mask_m_2 = bench.get_temp_buffer('col_t_alive_mask_m_2', colsel2_i)
        per_item_condition_not_satisfied_2 = bench.get_temp_buffer('per_item_condition_not_satisfied_2', colsel2_i)
        per_t_in_item_top_k_for2 = bench.get_temp_buffer('per_t_in_item_top_k_for2', colsel2_i)
        print('per_item_col_real_2', per_item_col_real_2)
        print('per_item_condition_not_satisfied_2', per_item_condition_not_satisfied_2)
        print('per_t_in_item_top_k_for2', per_t_in_item_top_k_for2)

    col_t_alive_mask_final = bench.get_temp_buffer('col_t_alive_mask_final', index_layer)
    col_t_before_masked = bench.get_temp_buffer('col_t_before_masked', index_layer)
    col_t_after_masked = bench.get_temp_buffer('col_t_after_masked', index_layer)

    if not kwargs['perlin_colsel_mask_in_probs']:
        estimated_attention_score_after_mp0 = bench.get_temp_buffer('estimated_attention_score_after_mp0', index_layer)

    # TODO add case for self.benchmark
    t_dead_mask_before_colsel = bench.get_temp_buffer('t_dead_mask_before_colsel', index_layer)
    t_dead_mask_after_colsel = bench.get_temp_buffer('t_dead_mask_after_colsel', index_layer)


else:
    t_dead_mask = bench.get_temp_buffer('t_dead_mask', index_layer)

per_item_top_k = bench.get_temp_buffer('per_item_top_k', index_layer)
per_t_in_item_top_k = bench.get_temp_buffer('per_t_in_item_top_k', index_layer)
print('per_item_top_k', per_item_top_k)
print('per_t_in_item_top_k', per_t_in_item_top_k)

# partial_attention_mask after topk, colsel (before interp)
partial_attention_mask_before_interp = bench.get_temp_buffer('partial_attention_mask_before_interp', index_layer)
partial_attention_mask_after_interp = bench.get_temp_buffer('partial_attention_mask_after_interp', index_layer)

attention_scores_dense = bench.get_temp_buffer('attention_scores_dense', index_layer)
# original dense probs
attention_probs_dense = bench.get_temp_buffer('attention_probs_dense', index_layer)
# partial_attention_probs
partial_attention_scores = bench.get_temp_buffer('partial_attention_scores', index_layer)
partial_attention_probs = bench.get_temp_buffer('attention_matrix', index_layer)

# partial_context_layer (not mandatory)
partial_context_layer_1 = bench.get_temp_buffer('partial_context_layer_1', index_layer) # N, T, H*HID
average_context_layer = bench.get_temp_buffer('average_context_layer', index_layer) # N, T, H*HID
partial_context_layer_2 = bench.get_temp_buffer('partial_context_layer_2', index_layer) # N, T, H*HID

# finally outputed estimated_attention_probs
estimated_attention_probs_for_output = bench.get_temp_buffer('estimated_attention_probs_for_output', index_layer)
partial_context_layer = bench.get_temp_buffer('partial_context_layer', index_layer) # N, T, H*HID

# "sum_mask" : col_select_mask
# if kwargs['perlin_colsel']:
#     if kwargs['perlin_colsel_method'] == 'sum_mask':
#         col_select_mask = bench.get_temp_buffer('col_select_mask', index_layer)
#     # large_inx_mask
#     large_inx_mask = bench.get_temp_buffer('large_inx_mask', index_layer)
#     # col_sel_estimated_attention_probs
#     col_sel_estimated_attention_probs_bef_select = bench.get_temp_buffer('col_sel_estimated_attention_probs_bef_select', index_layer)
#     col_sel_estimated_attention_probs_selcol_filled = bench.get_temp_buffer('col_sel_estimated_attention_probs_selcol_filled', index_layer)

N, T = attention_mask.shape[0], attention_mask.shape[-1]
INDEX= 1

# def imsave(img: torch.Tensor, path):
#     plt.clf()
#     plt.imshow(img.cpu().numpy())
#     plt.colorbar()
#     plt.savefig(path, dpi=300)
#     print(f'saved {path}')

def imsave(t: torch.Tensor, path):
    global INDEX
    img = convert_to_colormap(t.cpu().numpy())
    # path = f"./plots/poc/test_resizing/{name}.png"
    path = os.path.join(root, (str(INDEX)+'_'+path))
    cv2.imwrite(path, img)
    print('processed', path)
    INDEX += 1

bool2int = lambda x: 1 if x else 0
r = bool2int(kwargs['perlin_colsel'])
m = kwargs['perlin_colsel_method']
p = bool2int(kwargs['perlin_colsel_mask_in_probs'])
h = kwargs["perlin_colsel_per_head_col"]

root = f'./saves/tests/test_perlin_col_sel_batch/colsel{r}'
if kwargs['perlin_colsel']:
    root += f'_{m}_mprobs{p}'
if h > -1:
    root += f'_perhead{h}'
os.makedirs(root, exist_ok=True)

index_batch = 3
index_head = 2

imsave(estimated_attention_score_init[index_batch], 'est_score_init.png')#
imsave(estimated_attention_probs_init[index_batch], 'est_probs_init.png')#

imsave(estimated_attention_score_resized[index_batch], 'est_score_resized.png')
imsave(estimated_attention_probs_resized[index_batch], 'est_probs_resized.png')

imsave(attention_scores_truth[index_batch], 'scores_truth.png')
imsave(attention_probs_truth[index_batch], 'probs_truth.png')
imsave(attention_probs_truth_m[index_batch], 'probs_truth_m.png')

imsave(masked_estimated_attention_probs[index_batch], 'masked_est_probs.png')#


if kwargs['perlin_colsel']:
    imsave(col_mean_in_flatten, 'col_mean_in_flatten.png')
    imsave(col_result_mask.float(), 'col_result_mask.png')
    
    imsave(col_t_alive_mask_bef_inter_1[index_batch].float(), 'col_t_alive_mask_bef_inter_1.png')
    imsave(col_t_alive_mask_t_1[index_batch], 'col_t_alive_mask_t_1.png')
    imsave(col_t_alive_mask_m_1[index_batch], 'col_t_alive_mask_m_1.png')

    if colsel_perform_cnt>0:
        imsave(col_t_alive_mask_bef_inter_1_for2[index_batch].float(), 'col_t_alive_mask_bef_inter_1_for2.png')
        imsave(col_t_alive_mask_t_1_for2[index_batch], 'col_t_alive_mask_t_1_for2.png')
        imsave(col_t_alive_mask_m_1_for2[index_batch], 'col_t_alive_mask_m_1_for2.png')

        imsave(col_t_alive_mask_bef_inter_2[index_batch].float(), 'col_t_alive_mask_bef_inter_2.png')
        imsave(col_t_alive_mask_t_2[index_batch], 'col_t_alive_mask_t_2.png')
        imsave(col_t_alive_mask_m_2[index_batch], 'col_t_alive_mask_m_2.png')
    
    imsave(col_t_alive_mask_final[index_batch], 'col_t_alive_mask_final.png')#
    imsave(col_t_before_masked[index_batch], 'col_t_before_masked.png')
    imsave(col_t_after_masked[index_batch], 'col_t_after_masked.png')

    if not kwargs['perlin_colsel_mask_in_probs']:
        imsave(estimated_attention_score_after_mp0[index_batch], 'estimated_attention_score_after_mp0.png')

    imsave(t_dead_mask_before_colsel[index_batch], 't_dead_mask_before_colsel.png')
    imsave(t_dead_mask_after_colsel[index_batch], 't_dead_mask_after_colsel.png')
    
    # if kwargs['perlin_colsel_method'] == 'sum_mask':
    #     imsave(col_select_mask[index_batch], 'col_select_mask.png')

    # imsave(large_inx_mask[index_batch], 'large_inx_mask.png')
    # imsave(col_sel_estimated_attention_probs_bef_select[index_batch], 'colsel_est_probs_bef_select.png')

    # imsave(col_sel_estimated_attention_probs_selcol_filled[index_batch], 'colsel_est_probs_aft_select.png')
else:
    imsave(t_dead_mask[index_batch], 't_dead_mask.png')

imsave(partial_attention_mask_before_interp[index_batch], 'part_attn_mask_bef_interp.png')
imsave(partial_attention_mask_after_interp[index_batch], 'part_attn_mask_aft_interp.png')

imsave(attention_scores_dense[index_batch], 'attn_dense.png')
imsave(attention_probs_dense[index_batch], 'attn_dense.png')
imsave(partial_attention_scores[index_batch], 'part_attn_scores.png')
imsave(partial_attention_probs[index_batch], 'part_attn_probs.png')

imsave(partial_context_layer_1[index_batch, index_head], 'part_context_layer_1.png')
imsave(average_context_layer[index_batch, index_head], 'avg_context_layer.png')
imsave(partial_context_layer_2[index_batch, index_head], 'part_context_layer_2.png')

imsave(estimated_attention_probs_for_output[index_batch], 'est_probs_output.png')
imsave(partial_context_layer[index_batch], 'partial_context_layer.png')

# imsave(
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)) /\
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)).sum(dim=-1, keepdim=True), os.path.join(root, 'est_interp_norm.png'))
# imsave(partial_attention_mask_before_interp[index_batch,index_head], os.path.join(root, 'part.png'))
# imsave(partial_attention_mask[index_batch,index_head], os.path.join(root, 'part_interp.png'))
# imsave(attention_mask[index_batch,0].expand(T, T), os.path.join(root, 'attn_mask.png'))
# imsave(partial_attention_probs[index_batch,0].expand(T, T), os.path.join(root, 'final_partial_probs.png'))