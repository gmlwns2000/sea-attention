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

index_layer = 0

v_for_atten_identity_interpolate = bench.get_temp_buffer('v_for_atten_identity_interpolate', index_layer)
v_for_atten_identity_bef_grid = bench.get_temp_buffer('v_for_atten_identity_bef_grid', index_layer)
v_for_atten_identity_aft_grid = bench.get_temp_buffer('v_for_atten_identity_aft_grid', index_layer)
performer_context_layer = bench.get_temp_buffer('performer_context_layer', index_layer)
performer_context_layer_mask = bench.get_temp_buffer('performer_context_layer>0', index_layer)


def imsave(img: torch.Tensor, path):
    plt.clf()
    plt.imshow(img.cpu().numpy())
    plt.colorbar()
    plt.savefig(path, dpi=300)
    print(f'saved {path}')

root = './saves/tests/test_perlin_performer/'
os.makedirs(root, exist_ok=True)

index_batch = 1
index_head = 0

imsave(v_for_atten_identity_interpolate[index_batch,index_head], os.path.join(root, 'interpolate.png'))
imsave(v_for_atten_identity_bef_grid[index_batch,index_head], os.path.join(root, 'bef_grid.png'))
imsave(v_for_atten_identity_aft_grid[index_batch,index_head], os.path.join(root, 'aft_grid.png'))
imsave(performer_context_layer[index_batch,index_head], os.path.join(root, 'performer_context_layer.png'))
imsave(performer_context_layer_mask[index_batch,index_head], os.path.join(root, 'performer_context_layer>0.png'))