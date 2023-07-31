import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import torch
from .common_opt import init
from ...models import perlin_attention

def main():
    use_cache = True
    
    trainer, model, tokenizer = init(skip_init_loaders=True)
    
    input_ids = tokenizer(
        "Famitsu enjoyed the story , and were particularly pleased with the improvements to gameplay . Japanese gaming site Game Watch <unk> , despite negatively noting its pacing and elements recycled from previous games , was generally positive about its story and characters , and found its gameplay entertaining despite off @-@ putting difficulty spikes . <unk> writer <unk> <unk> , in a Play Test article based on the game 's <unk> demo , felt that Valkyria Chronicles III provided a profound feeling of closure for the Valkyria Chronicles series . He praised its gameplay despite annoying limitations to aspects such as special abilities , and positively noted its shift in story to a tone similar to the first game . PlayStation Official Magazine - UK praised the story 's <unk> of Gallia 's moral standing , art style , and most points about its gameplay , positively noting the latter for both its continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the previous games . Heath Hindman of gaming website PlayStation <unk> praised the addition of non @-@ linear elements and improvements or removal of mechanics from Valkyria Chronicles II in addition to praising the returning gameplay style of previous games . He also positively noted the story 's serious tone . Points criticized in the review were recycled elements , awkward cutscenes that seemed to include all characters in a scene for no good reason , pacing issues , and occasional problems with the game 's AI ",
        return_tensors="pt"
    ).input_ids.to(trainer.device) # type: torch.Tensor
    
    with torch.no_grad():
        output = model(input_ids)
    dense_output = torch.argmax(output.logits, dim=-1)[0].cpu().numpy()
    dense_text = tokenizer.batch_decode(dense_output.reshape(1, -1), skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(dense_output, dense_text)
    
    past_key_values = None
    output_ids = []
    perlin_attention.get_default_config().use_cache = use_cache
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
        else:
            ids_slice = input_ids[:, :i+1]
            with torch.no_grad():
                output = model(
                    input_ids=ids_slice,
                    use_cache=False
                )
            token_id = torch.argmax(output.logits[:,-1,:], dim=-1).item()
        output_ids.append(token_id)
    cached_output = np.array(output_ids)
    cached_text = tokenizer.batch_decode(cached_output.reshape(1, -1), skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(cached_output, cached_text)
    print('accuracy', ((cached_output == dense_output) * 1.0).mean())

if __name__ == '__main__':
    main()