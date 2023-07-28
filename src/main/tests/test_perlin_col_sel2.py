from ...utils import get_bench
from ..visualize_glue import main as visualize_main
from ..visualize_glue import add_perlin_model_options, parse_perlin_model_options
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

# index_layer = 7

# estimated_attention_probs = bench.get_temp_buffer('estimated_attention_probs', index_layer)
# attention_probs_dense = bench.get_temp_buffer('attention_probs_dense', index_layer)
# estimated_attention_probs_for_output = bench.get_temp_buffer('estimated_attention_probs_for_output', index_layer)
# partial_attention_mask_before_interp = bench.get_temp_buffer('partial_attention_mask_before_interp', index_layer)
# partial_attention_mask = bench.get_temp_buffer('partial_attention_mask', index_layer)
# attention_mask = bench.get_temp_buffer('attention_mask', index_layer)
# partial_attention_probs = bench.get_temp_buffer('attention_matrix', index_layer)

# N, T = attention_mask.shape[0], attention_mask.shape[-1]

# def imsave(img: torch.Tensor, path):
#     plt.clf()
#     plt.imshow(img.cpu().numpy())
#     plt.colorbar()
#     plt.savefig(path, dpi=300)
#     print(f'saved {path}')

# root = './saves/tests/test_perlin_est_atten/'
# os.makedirs(root, exist_ok=True)

# index_batch = 3
# index_head = 2

# imsave(estimated_attention_probs[index_batch,index_head], os.path.join(root, 'est.png'))
# imsave(attention_probs_dense[index_batch,index_head], os.path.join(root, 'dense_origin.png'))
# imsave(estimated_attention_probs_for_output[index_batch,index_head], os.path.join(root, 'est_interp.png'))
# imsave(
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)) /\
#     (estimated_attention_probs_for_output[index_batch,index_head] * (attention_mask[index_batch,0] > -1)).sum(dim=-1, keepdim=True), os.path.join(root, 'est_interp_norm.png'))
# imsave(partial_attention_mask_before_interp[index_batch,index_head], os.path.join(root, 'part.png'))
# imsave(partial_attention_mask[index_batch,index_head], os.path.join(root, 'part_interp.png'))
# imsave(attention_mask[index_batch,0].expand(T, T), os.path.join(root, 'attn_mask.png'))
# imsave(partial_attention_probs[index_batch,0].expand(T, T), os.path.join(root, 'final_partial_probs.png'))

