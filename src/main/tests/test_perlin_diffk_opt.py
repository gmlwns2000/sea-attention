# load k13

# k 3~65 stepping with 3 increased

# 1. get accuracy
    
import warnings
from ...models import perlin_bert
from transformers import AutoConfig, AutoTokenizer
from ...dataset.wikitext2 import get_dataloader
from ...utils import batch_to
import torch, tqdm, argparse
from ...trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options
from ...trainer.perlin_trainer import OptTrainer
from ...models import perlin_attention
import os

def main(
    subset = 'wikitext2',
    checkpoint_path = None,
    evaluate = False,
    **kwargs,
):
    k_test=[]
    for i in range(31):
        k_test.append(16+8*i)
    print(f'k_test : {k_test}')    
    for trained_k in [64]:# 32, 64, 
        warnings.warn(f"trained k {trained_k}")
        kwargs['perlin_k']=trained_k
        print(kwargs)
        
        trainer = OptTrainer(
            model='opt-125m',
            subset='wikitext2',
            **kwargs
        )
        trainer.load(path=checkpoint_path)
        
        trainer.base_model.eval()
        trainer.model.eval()
        
        for module in trainer.model.modules():
            if hasattr(module, 'benchmarking'):
                module.benchmarking = False
        
        # valid_loader = get_dataloader(
        #     subset='valid', 
        #     tokenizer=trainer.tokenizer, 
        #     batch_size=1, 
        #     max_length=32000,
        # )
        perlin_attention.get_default_config().k = trained_k
        pretrained_ppl = trainer.evaluate()
        print(f'pretrained_k:{trained_k}, pretrained_ppl:{pretrained_ppl}')
        
        k_acc_bucket={}
        for k in k_test:
            perlin_attention.get_default_config().k = k
            score = trainer.evaluate()
            # acc_sum = 0
            # acc_count = 0
            # with tqdm.tqdm(valid_loader, dynamic_ncols=True) as pbar:
            #     for batch in pbar:
            #         batch = batch_to(batch, trainer.device)
            #         with torch.no_grad(), torch.autocast('cuda', torch.float32):
            #             batch['output_attentions'] = True
            #             batch['output_hidden_states'] = True
            #             trainer.base_model(**batch)
            #             batch['teacher'] = trainer.base_model
            #             output = trainer.model(**batch)
            #             acc_sum += (torch.argmax(output.logits, dim=-1) == batch['labels']).float().sum().item()
            #             acc_count += len(batch['labels'])
                    
            #         pbar.set_description(f'k:{k}, acc:{acc_sum/acc_count:.4f}')
            # final_acc = acc_sum/acc_count*100
            
            print(f'k:{k}, ppl:{score}')
            k_acc_bucket[k]=score
        print(f'finalized k_acc_bucket \n {k_acc_bucket}')        
        os.makedirs(f'./saves/tests/k_acc_bucket/', exist_ok=True)
        torch.save({'pretrained_k':trained_k, 'pretrained_ppl': pretrained_ppl,'k_acc_bucket':k_acc_bucket}, f'./saves/tests/k_acc_bucket/opt/k_{trained_k}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    add_perlin_model_options(parser)

    args = parser.parse_args()
    # print(args)

    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'checkpoint_path': args.checkpoint,
        'evaluate': args.evaluate
    })

    main(**kwargs)