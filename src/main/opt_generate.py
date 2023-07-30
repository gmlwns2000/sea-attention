import gc
import time
import torch
import cv2
import os
import argparse

import tqdm

from ..utils import batch_to
from ..models import perlin_opt
from ..models import perlin_attention
from ..trainer.perlin_trainer import OptTrainer, add_perlin_model_options, parse_perlin_model_options
from transformers import OPTForCausalLM

def main(
    dataset = 'wikitext2',
    checkpoint_path = None,
    **kwargs
):
    trainer = OptTrainer(
        model='opt',
        subset=dataset,
        **kwargs,
    )
    if checkpoint_path is None:
        checkpoint_path = trainer.checkpoint_path()
    if os.path.exists(checkpoint_path):
        trainer.load(path=checkpoint_path)
    else:
        print('checkpoint not exists', checkpoint_path)
    
    model = trainer.model.eval() # type: perlin_opt.OPTForCausalLM
    tokenizer = trainer.tokenizer
    
    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids.to(trainer.device), max_length=kwargs['max_seq_len'])
        generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return inputs, generate_ids, generated_text
    
    perlin_attention.get_default_config().use_cache = True
    sample_text = "Robert <unk> is an English film , television and theatre actor ."
    _, _, generated_text = generate(sample_text)
    print('sample:', sample_text)
    print('generated:', generated_text)
    
    while True:
        print('>>> ', end='', flush=True)
        prompt = input().strip()
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        
        t = time.time()
        inputs, generate_ids, generated_text = generate(prompt)
        end_mem = torch.cuda.max_memory_allocated()
        elapsed = time.time() - t
        
        print(generate_ids)
        print(f"```{generated_text.strip()}```")
        print(f'elapsed: {elapsed*1000:.2f} ms, {(generate_ids.shape[-1] - inputs.input_ids.shape[-1]) / elapsed:.2f} wps, {(end_mem - start_mem) / 1024 / 1024:.2f} MB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--max-seq-len', type=int, default=768)
    add_perlin_model_options(parser)
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'dataset': args.dataset,
        'checkpoint_path': args.checkpoint,
        'max_seq_len': args.max_seq_len,
    })
    
    main(**kwargs)