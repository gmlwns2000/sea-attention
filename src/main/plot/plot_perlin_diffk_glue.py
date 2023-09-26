# x: different k values
# y: accuracy    
import warnings
from matplotlib import pyplot as plt
import torch
from ...utils import seed
import random
import argparse
from ...trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options
from ...trainer.perlin_trainer import GlueTrainer
import os

plt.style.use('seaborn-bright')

def load_and_plot(
    subset = 'mnli',
    checkpoint_path = None,
    evaluate = False,
    **kwargs,
):  
    model = kwargs['model']
    if model=='bert':
        trained_k = [7, 13, 25]
    elif 'opt' in model:
        trained_k = [32, 64, 128]
        
    def plot(filename, title, ylabel):
        plt.clf()
        
        for itk, tk in enumerate(trained_k):
            kwargs['perlin_k'] = tk
            trainer = GlueTrainer(
                    subset=subset,
                    **kwargs
                )
            path = f'./saves/tests/k_acc_bucket/k_{tk}.pth'
            state = torch.load(path, map_location='cpu')
            k_acc_bucket = state['k_acc_bucket'] # dict with k:acc form
            del state
            
            print(f'test_k {tk}')
            print(f'k_acc_bucket {k_acc_bucket}')
            ks = list(k_acc_bucket.keys())
            metrics = list(k_acc_bucket.values())
            plt.plot(
                ks, 
                metrics, 
                label=f'k={tk}', 
                linestyle='--', 
                linewidth=0.75,
                marker='*',
            )
        
        plt.title(f'{title}')
        plt.xlabel(f'Hyperparameter k')
        plt.ylabel(f'{ylabel}')
        plt.grid()
        plt.legend(fontsize=8)
        
        root = f'./plots/main/eval_diffk_{model}'
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f'{filename}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print('saved', path)
        path = os.path.join(root, f'{filename}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print('saved', path)
    
    plot(
        f'{model}_diffk', 
        'Test time performance of adjusted k  (Glue-MNLI)', 'Acc. \u2191', 
    )            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--subset', type=str, default='mnli')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    add_perlin_model_options(parser)

    args = parser.parse_args()
    print(args)

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'model': args.model,
        'subset': args.subset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate
    })
    load_and_plot(**kwargs)