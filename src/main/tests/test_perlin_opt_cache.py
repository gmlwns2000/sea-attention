import os, tqdm, gc
import flax
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import torch
from .common_opt import init
from ...models import perlin_attention
from ...utils import get_bench, strify
import torch.nn.functional as F

TRACKING_BUFFERS = [
    'q',
    'k',
    'v_for_atten',
    'performer_value',
    'performer_context_layer',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_context_layer',
]

BUFFER_ACCUMULATE = {
    'q', 
    'performer_value',
    'performer_context_layer',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_context_layer',
    'logits',
}

INDEX_LAYER = 0
MAX_SEQ_LEN = 64

def main():
    use_cache = True
    bench = get_bench()
    bench.activate_temp_buffers = True
    
    trainer, model, tokenizer = init(skip_init_loaders=True, checkpoint_path='./saves/trainer/opt_trainer/opt-125m_wikitext2_kf1_lw0_perlin_k64_full_copy/checkpoint.pth')
    
    input_ids = tokenizer(
        "Famitsu enjoyed the story , and were particularly pleased with the improvements to gameplay . Japanese gaming site Game Watch <unk> , despite negatively noting its pacing and elements recycled from previous games , was generally positive about its story and characters , and found its gameplay entertaining despite off @-@ putting difficulty spikes . <unk> writer <unk> <unk> , in a Play Test article based on the game 's <unk> demo , felt that Valkyria Chronicles III provided a profound feeling of closure for the Valkyria Chronicles series . He praised its gameplay despite annoying limitations to aspects such as special abilities , and positively noted its shift in story to a tone similar to the first game . PlayStation Official Magazine - UK praised the story 's <unk> of Gallia 's moral standing , art style , and most points about its gameplay , positively noting the latter for both its continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the previous games . Heath Hindman of gaming website PlayStation <unk> praised the addition of non @-@ linear elements and improvements or removal of mechanics from Valkyria Chronicles II in addition to praising the returning gameplay style of previous games . He also positively noted the story 's serious tone . Points criticized in the review were recycled elements , awkward cutscenes that seemed to include all characters in a scene for no good reason , pacing issues , and occasional problems with the game 's AI ",
        return_tensors="pt"
    ).input_ids.to(trainer.device) # type: torch.Tensor
    input_ids = input_ids[:,:min(input_ids.shape[-1], MAX_SEQ_LEN)]
    
    # sample dense
    with torch.no_grad():
        output = model(input_ids)
    
    buffers_truth = {}
    for name in TRACKING_BUFFERS:
        # sample only first layer
        buffers_truth[name] = bench.get_temp_buffer(name, index=INDEX_LAYER)
    buffers_truth['logits'] = output.logits
    bench.reset_temp_buffers()
    
    dense_output = torch.argmax(output.logits, dim=-1)[0].cpu().numpy()
    dense_text = tokenizer.batch_decode(dense_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(dense_output, dense_text)
    
    # sample with cache
    past_key_values = None
    output_ids = []
    perlin_attention.get_default_config().use_cache = use_cache
    buffers = {}
    for i in range(input_ids.shape[-1]):
        if use_cache:
            ids_slice = input_ids[:, i:i+1]
            with torch.no_grad():
                output = model(
                    input_ids=ids_slice,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = output.past_key_values
            token_id = torch.argmax(output.logits, dim=-1).item()
            
            for name in TRACKING_BUFFERS:
                buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                if not name in buffers:
                    buffers[name] = buf
                else:
                    if name in BUFFER_ACCUMULATE:
                        buffers[name] = torch.cat([buffers[name], buf], dim=-2)
                    else:
                        buffers[name] = buf
            
            buf = output.logits
            if not 'logits' in buffers:
                buffers['logits'] = buf
            else:
                if 'logits' in BUFFER_ACCUMULATE:
                    buffers['logits'] = torch.cat([buffers['logits'], buf], dim=-2)
                else:
                    buffers['logits'] = buf
            
            bench.reset_temp_buffers()
        else:
            ids_slice = input_ids[:, :i+1]
            with torch.no_grad():
                output = model(
                    input_ids=ids_slice,
                    use_cache=False
                )
            token_id = torch.argmax(output.logits[:,-1,:], dim=-1).item()
            
            for name in TRACKING_BUFFERS:
                buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                buffers[name] = buf
            buffers['logits'] = output.logits
            bench.reset_temp_buffers()
        output_ids.append(token_id)
        
    cached_output = np.array(output_ids)
    cached_text = tokenizer.batch_decode(cached_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(cached_output, cached_text)
    print('accuracy', ((cached_output == dense_output) * 1.0).mean())
    
    print('truth', strify(buffers_truth))
    print('buffers', strify(buffers))
    
    CHECK_INDEX = [0, 1, 2, 10, 20, 30, 40]
    JUST_WIDTH = 12
    for name in TRACKING_BUFFERS + ['logits']:
        print(f'- {name}')
        truth = buffers_truth[name]
        mine = buffers[name]
        losses = []
        for idx in CHECK_INDEX:
            loss = F.mse_loss(truth[...,idx,:], mine[...,idx,:], reduction='sum').item()
            losses.append(loss)
        print(f'  INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
        print(f'  ERROR: {",".join([f"{loss:.4f}".rjust(JUST_WIDTH) for loss in losses])}')

if __name__ == '__main__':
    main()