from ...utils import batch_to
from ...models.dict.model_self_attention import get_model_self_attn
import torch
from ...dataset.test_batch_save_load import load_test_batch

def sample_attentions_model(
        device,
        for_eval,
        test_batch_size,
        base_model_type, 
        dataset, 
        subset, 
        model, 
        base_model, 
        viz_dense_attn = True):
    
    test_batch = load_test_batch(dataset, subset, for_eval, test_batch_size)
    test_batch = batch_to(test_batch, device)
    
    test_batch['output_hidden_states'] = True
    test_batch['output_attentions'] = True

    with torch.no_grad():
        output_base = base_model(**test_batch)
    test_batch['teacher'] = base_model # NOTE attention_method =='base' doesn't call this func

    with torch.no_grad():
        model(**test_batch)

    all_layers_attn_probs = []
    model_self_attention = get_model_self_attn(base_model_type)
    for module in model.modules():
        if isinstance(module, model_self_attention):
            if viz_dense_attn:
                assert module.last_dense_attention_prob is not None
                all_layers_attn_probs.append(module.last_dense_attention_prob)
            if not viz_dense_attn:
                assert module.last_sparse_attention_prob is not None
                all_layers_attn_probs.append(module.last_sparse_attention_prob)
    return all_layers_attn_probs

def sample_attentions_basem(
        device,
        for_eval,
        test_batch_size,
        dataset, 
        subset, 
        base_model):
    
    test_batch = load_test_batch(dataset, subset, for_eval, test_batch_size)
    test_batch = batch_to(test_batch, device)

    test_batch['output_hidden_states'] = True
    test_batch['output_attentions'] = True

    with torch.no_grad():
        output_base = base_model(**test_batch)

    all_layers_attn_probs = output_base.attentions # [Tuple[torch.FloatTensor]]
    all_layers_attn_probs = list(all_layers_attn_probs)

    return all_layers_attn_probs