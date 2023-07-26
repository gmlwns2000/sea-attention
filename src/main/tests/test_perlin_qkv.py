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

q = bench.get_temp_buffer('q', index_layer)
k = bench.get_temp_buffer('k', index_layer)
v = bench.get_temp_buffer('v', index_layer)
q_for_atten = bench.get_temp_buffer('q_for_atten', index_layer)
k_for_atten = bench.get_temp_buffer('k_for_atten', index_layer)
v_for_atten = bench.get_temp_buffer('v_for_atten', index_layer)
q_for_score = bench.get_temp_buffer('q_for_score', index_layer)
k_for_score = bench.get_temp_buffer('k_for_score', index_layer)

def imsave(img: torch.Tensor, path):
    plt.clf()
    plt.imshow(img.cpu().numpy())
    plt.colorbar()
    plt.savefig(path, dpi=300)
    print(f'saved {path}')

root = './saves/tests/test_perlin_qkv/'
os.makedirs(root, exist_ok=True)

index_batch = 3
index_head = 0

imsave(q[index_batch,index_head], os.path.join(root, 'q.png'))
imsave(k[index_batch,index_head], os.path.join(root, 'k.png'))
imsave(v[index_batch,index_head], os.path.join(root, 'v.png'))
imsave(q_for_atten[index_batch,index_head], os.path.join(root, 'q_for_atten.png'))
imsave(k_for_atten[index_batch,index_head], os.path.join(root, 'k_for_atten.png'))
imsave(v_for_atten[index_batch,index_head], os.path.join(root, 'v_for_atten.png'))
imsave(q_for_score[index_batch,index_head], os.path.join(root, 'q_for_score.png'))
imsave(k_for_score[index_batch,index_head], os.path.join(root, 'k_for_score.png'))
