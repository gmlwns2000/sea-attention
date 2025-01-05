# load k13

# k 3~65 stepping with 3 increased

# 1. get accuracy
    
import warnings
from ...models import perlin_bert
from transformers import AutoConfig, AutoTokenizer
from ...dataset.glue import get_dataloader, TASK_TO_VALID
from ...utils import batch_to
import torch, tqdm, argparse
from ...trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options
from ...trainer.perlin_trainer import GlueTrainer
from ...models import perlin_attention
import os

def main(
    subset = 'mnli',
    checkpoint_path = None,
    evaluate = False,
    **kwargs,
):
    k_test=[]
    for i in range(21):
        k_test.append(3+3*i)
    print(f'k_test : {k_test}')
    for trained_k in [7,13,25]:
        warnings.warn(f"trained k {trained_k}")
        kwargs['perlin_k']=trained_k
        print(kwargs)
        
        trainer = GlueTrainer(
            subset=subset,
            **kwargs
        )
        trainer.load(path=checkpoint_path)
        
        trainer.base_model.eval()
        trainer.model.eval()
        
        for module in trainer.model.modules():
            if hasattr(module, 'benchmarking'):
                module.benchmarking = False
        
        batch_size = 16
        encode_batch_size = 384
        valid_loader = get_dataloader(
            trainer.subset, 
            trainer.tokenizer, 
            batch_size, 
            TASK_TO_VALID[trainer.subset], 
            encode_batch_size
        )
        acc_sum = 0
        acc_count = 0
        perlin_attention.get_default_config().k=trained_k
        with tqdm.tqdm(valid_loader, dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch = batch_to(batch, trainer.device)
                with torch.no_grad(), torch.autocast('cuda', torch.float32):
                    batch['output_attentions'] = True
                    batch['output_hidden_states'] = True
                    trainer.base_model(**batch)
                    batch['teacher'] = trainer.base_model
                    output = trainer.model(**batch)
                    acc_sum += (torch.argmax(output.logits, dim=-1) == batch['labels']).float().sum().item()
                    acc_count += len(batch['labels'])

                pbar.set_description(f'k:{trained_k}, acc:{acc_sum/acc_count:.4f}')
        pretrained_acc = acc_sum/acc_count*100
        print(f'pretrained_k:{trained_k}, accuracy:{pretrained_acc} %')

        k_acc_bucket={}
        for k in k_test:
            acc_sum = 0
            acc_count = 0

            perlin_attention.get_default_config().k = k

            with tqdm.tqdm(valid_loader, dynamic_ncols=True) as pbar:
                for batch in pbar:
                    batch = batch_to(batch, trainer.device)
                    with torch.no_grad(), torch.autocast('cuda', torch.float32):
                        batch['output_attentions'] = True
                        batch['output_hidden_states'] = True
                        trainer.base_model(**batch)
                        batch['teacher'] = trainer.base_model
                        output = trainer.model(**batch)
                        acc_sum += (torch.argmax(output.logits, dim=-1) == batch['labels']).float().sum().item()
                        acc_count += len(batch['labels'])
                    
                    pbar.set_description(f'k:{k}, acc:{acc_sum/acc_count:.4f}')
            final_acc = acc_sum/acc_count*100
            print(f'k:{k}, dataset accuracy:{final_acc} %')
            k_acc_bucket[k]=final_acc
        print(f'finalized k_acc_bucket \n {k_acc_bucket}')
        os.makedirs(f'./saves/tests/k_acc_bucket2/', exist_ok=True)
        torch.save({'pretrained_k':trained_k, 'pretrained_accuracy':pretrained_acc,'k_acc_bucket':k_acc_bucket}, f'./saves/tests/k_acc_bucket2/k_{trained_k}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--subset', type=str, default='mnli')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    add_perlin_model_options(parser)

    args = parser.parse_args()
    # print(args)

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'subset': args.subset,
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate
    })

    main(**kwargs)