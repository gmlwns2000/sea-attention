from ...utils import get_bench
from ..visualize.glue import main as visualize_main
from ..visualize.glue import add_perlin_model_options, parse_perlin_model_options
import argparse, os, torch
import matplotlib.pyplot as plt
from ..visualize.common import convert_to_colormap
import cv2

# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()
#     torch.multiprocessing.set_start_method('spawn') #, force=True
    
BENCHMARK = True
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
    'evaluate': args.evaluate,
    'benchmark_for_sparse' : BENCHMARK
})

visualize_main(**kwargs)

index_layer = 11

attention_mask = bench.get_temp_buffer('attention_mask', index_layer)

if kwargs['perlin_colsel']:
    colsel_perform_cnt = bench.get_temp_buffer('colsel_perform_cnt', index_layer)
    print('colsel_perform_cnt', colsel_perform_cnt)

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

# TODO unlock after error in attention is fixed
partial_attention_mask_bef_coo_intp = bench.get_temp_buffer('partial_attention_mask_bef_coo_intp', index_layer)
partial_attention_mask_aft_coo_intp = bench.get_temp_buffer('partial_attention_mask_aft_coo_intp', index_layer)

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

root = f'./saves/tests/test_perlin_coo_batch/colsel{r}'
if kwargs['perlin_colsel']:
    root += f'_{m}_mprobs{p}'
if h > -1:
    root += f'_perhead{h}'
os.makedirs(root, exist_ok=True)

index_batch = 3
index_head = 2

if kwargs['perlin_colsel']:
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

imsave(partial_attention_mask_bef_coo_intp[index_batch], 'partial_attention_mask_bef_coo_intp.png')
imsave(partial_attention_mask_aft_coo_intp[index_batch], 'partial_attention_mask_aft_coo_intp.png')