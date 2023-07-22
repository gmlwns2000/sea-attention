from ...models import perlin_bert
from transformers import AutoConfig, AutoTokenizer
from ...dataset.glue import get_dataloader, TASK_TO_VALID
from ...utils import batch_to
import torch, tqdm, argparse
from ...trainer.perlin_trainer import add_perlin_model_options, parse_perlin_model_options
from ...trainer.perlin_trainer import GlueTrainer

def main(
    subset = 'mnli',
    checkpoint_path = None,
    evaluate = False,
    **kwargs
):
    trainer = GlueTrainer(
        subset=subset,
        **kwargs
    )
    trainer.load(path=checkpoint_path)
    
    trainer.base_model.eval()
    trainer.model.eval()
    
    acc_sum = 0
    acc_count = 0
    k_sum = 0
    k_count = 0
    with tqdm.tqdm(trainer.valid_loader, dynamic_ncols=True) as pbar:
        for batch in pbar:
            batch = batch_to(batch, trainer.device)
            with torch.no_grad(), torch.autocast('cuda', torch.float16):
                batch['output_attentions'] = True
                batch['output_hidden_states'] = True
                trainer.base_model(**batch)
                batch['teacher'] = trainer.base_model
                output = trainer.model(**batch)
                acc_sum += (torch.argmax(output.logits, dim=-1) == batch['labels']).float().sum().item()
                acc_count += len(batch['labels'])
            
            for layer in trainer.model.bert.encoder.layer:
                layer = layer # type: perlin_bert.BertLayer
                N, H, T, T_1 = layer.attention.self.last_perlin_partial_probs.shape
                assert T == T_1
                elem_alive = (layer.attention.self.last_perlin_partial_probs.abs() > 1e-8).float().view(N, -1).sum(dim=-1)
                token_length = batch['attention_mask'].sum(-1)
                k = (elem_alive / token_length / H).sum()
                k_sum += k.item()
                k_count += N
            pbar.set_description(f'k:{k_sum/k_count:.2f} acc:{acc_sum/acc_count:.4f}')
    print(f'dataset average k = {k_sum / k_count:.3f}, dataset accuracy = {acc_sum / acc_count * 100:.2f} %')

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

    main(**kwargs)