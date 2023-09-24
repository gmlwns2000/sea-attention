# x: different k values
# y: accuracy    
from matplotlib import pyplot as plt
import torch
from ..utils import seed

plt.style.use('seaborn-bright')

k_acc_bucket = {}
seed()
for i in range(21):
    k_acc_bucket[3+3*i] = torch.randn*100
print(f'k_acc_bucket {k_acc_bucket}')

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
    for t_k in trained_k:
        kwargs['perlin_k'] = t_k
        trainer = GlueTrainer(
                subset=subset,
                **kwargs
            )
        path = f'./saves/tests/k_acc_bucket/{trainer.checkpoint_path}/k_{trained_k}.pth'
        # state = torch.load(path)
        # k_acc_bucket = state['k_acc_bucket'] # dict with k:acc form

        ks = list(k_acc_bucket.keys())
        metrics = list(k_acc_bucket.values())
        
        def plot(filename, title, ylabel, trained_k, ks, metrics):
            plt.clf()
            
            MARKERS = {
                'none': '>',
                'performer': 'v',
                'reformer': '^',
                'scatterbrain': 'x',
                'sinkhorn': 'h',
                'synthesizer': '.',
            }
            
            for itk, tks in enumerate(trained_k):
                plt.plot(
                    ks, 
                    metrics, 
                    label=f'k={tks}', 
                    linestyle='--', 
                    linewidth=0.75,
                    marker='*',
                )
            # for ik, k in enumerate(ks):
            #     plt.plot(
            #         ts, 
            #         perlins[ik], 
            #         label=f'Ours (k={k})', 
            #         linewidth=0.75,
            #         marker='*',
            #     )
            
            plt.title(f'{title}')
            plt.xlabel(f'hyperparameter k')
            plt.ylabel(f'{ylabel}')
            plt.grid()
            plt.legend(fontsize=8)
            
            root = './plots/main/eval_diffk_bert'
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, f'{filename}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print('saved', path)
            path = os.path.join(root, f'{filename}.pdf')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print('saved', path)
        
        plot(
            f'{model}_test_diffk', 
            'Evaluation with different k (SEA)', 'accuracy (%)', 
            ks, 
            metrics,
        )            

if __name__ == '__main__':
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

    load_and_plot(**kwargs)