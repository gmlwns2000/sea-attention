import os, tqdm, gc
import flax
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import torch
from .common_opt import init
from ...models import perlin_attention
from ...utils import get_bench, strify
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

TRACKING_BUFFERS = [
    'q',
    'k',
    'v_for_atten',
    'performer_context_layer',
    'performer_value',
    't_attention_predictor',
    'estimated_attention_score_dec_row',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
    'partial_attention_scores',
    'estimated_scales',
    'average_scale',
    'average_context_layer',
    'partial_context_layer_sparse',
    'normalized_partial_context_layer',
    'partial_context_layer',
]

BUFFER_ACCUMULATE = {
    'q', 
    'performer_context_layer',
    'performer_value',
    't_attention_predictor',
    'estimated_attention_score_dec_row',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_attention_scores',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
    'partial_context_layer',
    'estimated_scales',
    'average_scale',
    'average_context_layer',
    'partial_context_layer_sparse',
    'normalized_partial_context_layer',
    'logits',
}

DST_SOURCE_BUFFER = {
    'partial_attention_mask',
    'partial_attention_scores',
}

MASK_BUFFER = {
    'partial_attention_scores',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
}

INDEX_LAYER = 0
MAX_SEQ_LEN = 128

def main():
    use_cache = True
    bench = get_bench()
    bench.disabled = False
    bench.activate_temp_buffers = True
    
    # trainer, model, tokenizer = init(skip_init_loaders=True, checkpoint_path='./saves/trainer/opt_trainer/opt-125m_wikitext2_kf1_lw0_perlin_k64_full_copy/checkpoint.pth')
    trainer, model, tokenizer = init(skip_init_loaders=True)
    
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
        if name in bench.buffers:
            buffers_truth[name] = bench.get_temp_buffer(name, index=INDEX_LAYER)
    buffers_truth['logits'] = output.logits
    bench.reset_temp_buffers()
    
    dense_output = torch.argmax(output.logits, dim=-1)[0].cpu().numpy()
    dense_text = tokenizer.batch_decode(dense_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # print(dense_output, dense_text)
    
    # sample with cache
    past_key_values = None
    output_ids = []
    perlin_attention.get_default_config().use_cache = use_cache
    buffers = {}
    for i in tqdm.tqdm(range(input_ids.shape[-1])):
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
                if name in bench.buffers:
                    buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                    if not name in buffers:
                        buffers[name] = buf
                    else:
                        if name in BUFFER_ACCUMULATE:
                            if name in DST_SOURCE_BUFFER:
                                buffers[name] = F.pad(buffers[name], pad=(0, buf.shape[-1] - buffers[name].shape[-1]), mode='constant', value=-32000)
                            assert buffers[name].shape[-1] == buf.shape[-1], f"{name}: {buffers[name].shape}, {buf.shape}"
                            assert buffers[name].shape[:-2] == buf.shape[:-2], f"{name}: {buffers[name].shape}, {buf.shape}"
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
                if name in bench.buffers:
                    buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                    # buffers[name] = buf
                    if name in BUFFER_ACCUMULATE:
                        buf = buf[...,-1:,:]
                    if not name in buffers:
                        buffers[name] = buf
                    else:
                        if name in BUFFER_ACCUMULATE:
                            buffers[name] = torch.cat([buffers[name], buf], dim=-2)
                        else:
                            buffers[name] = buf
            
            # buffers['logits'] = output.logits
            buf = output.logits[...,-1:,:]
            if not 'logits' in buffers:
                buffers['logits'] = buf
            else:
                buffers['logits'] = torch.cat([buffers['logits'], buf], dim=-2)
            
            bench.reset_temp_buffers()
        output_ids.append(token_id)
        
    cached_output = np.array(output_ids)
    cached_text = tokenizer.batch_decode(cached_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # print(cached_output, cached_text)
    print('accuracy', ((cached_output == dense_output) * 1.0).mean())
    
    # print('truth', strify(buffers_truth))
    # print('buffers', strify(buffers))
    os.makedirs('./saves/tests/test_perlin_opt_cache', exist_ok=True)
    torch.save({
        'truth': buffers_truth, 
        'buffers': buffers
    }, './saves/tests/test_perlin_opt_cache/buf.pth')
    for name in buffers_truth.keys():
        assert buffers[name].shape == buffers_truth[name].shape, f"{name}: {buffers[name].shape} == {buffers_truth[name].shape}"
    
    def preproc(buffers):
        for name in buffers.keys():
            if name in MASK_BUFFER:
                buffers[name] = (buffers[name] > -1).float()
    preproc(buffers)
    preproc(buffers_truth)
    
    CHECK_INDEX = [0, 1, 2, 10, 20, 30, 40, -1]
    JUST_WIDTH = 12
    print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
    for name in TRACKING_BUFFERS + ['logits']:
        if name in buffers_truth and name in buffers:
            truth = buffers_truth[name]
            mine = buffers[name]
            losses = []
            for idx in CHECK_INDEX:
                def error(x, y):
                    x = x.to(torch.float64)
                    y = y.to(torch.float64)
                    return (x - y).abs().sum().log10()
                loss = error(truth[...,idx,:], mine[...,idx,:]).item()
                losses.append(loss)
            def deco_error(str, e):
                if e < -3:
                    return f"\033[92m{str}\033[0m"
                return f"\033[91m{str}\033[0m"
            print(f' - {name.ljust(JUST_WIDTH*3)} | ERROR: {",".join([deco_error(f"{loss:.4f}".rjust(JUST_WIDTH), loss) for loss in losses])}')

if __name__ == '__main__':
    main()