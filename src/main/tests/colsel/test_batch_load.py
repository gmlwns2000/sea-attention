# load saved ones and test values
'''
TODO list
** colwise 안하는 게 좋으면 안해야 ~ : max select 개수 선정하고, sum_mask에서 sum per col 특정 값 이상이어야만(Ti)<<애매 select하도록
layer당 다르게 적용됨, 그에 따라 batchwise k값 유지될 수도 있고, 더 적게 줄어들 수 있음 (selected_col_cnt 변동 가능)

** (sum_mask의 경우)실제 topk보다 masked_estimated_attention_probs 두꺼워서 발생할 수 있는 문제 (더 중요한 column이 select되지 않을 수 있음)
sum_per_col이 특정 값<< 이상일 시, 걔네끼리 큰 value 비교, max select 개수까지만 선택 (그게 topk 개수를 넘지 않을 경우 그 개수까지만 select)


1. masked_estimated_attention_probs 두께 확인, 
2. (sum_mask일시) col_select_mask 두께 확인,
3. sum_per_col 값 확인 -> 결과 확인, topk에서 동일한 혹은 적은 차이 나는 얘들이 무시되고 있을 수 있음
4. 

** 최종 select은 더 두껍게 조정? for interpolation : interpolation 전후 비교, interpolation 후에도 잘 남아있는지

1. selected column cnt is proper one
2. measure masked_estimated attn probs columns thinkness (TODO how to measure?)
3. masked_estimated attention probs

** 이 모든 걸 learnable? 하게 할 수 있는가?? - 현재는 loss 는 context layer에만 영향줌

'''
d = {
    'attention_mask.shape' : attention_mask.shape,
    'attention_mask': attention_mask,

    'estimated_attention_score.shape' : estimated_attention_score.shape,
    'estimated_attention_score': estimated_attention_score,

    'estimated_attention_probs.shape' : estimated_attention_probs.shape,
    'estimated_attention_probs_bef_masked': estimated_attention_probs,

    'masked_estimated_attention_probs.shape' : masked_estimated_attention_probs.shape,
    'masked_estimated_attention_probs': masked_estimated_attention_probs,

    'estimated_attention_score_resized.shape' : estimated_attention_score_resized.shape,
    'estimated_attention_score_resized': estimated_attention_score_resized,

    'estimated_attention_probs_resized.shape' : estimated_attention_probs_resized.shape,
    'estimated_attention_probs_resized': estimated_attention_probs_resized,

    'attention_probs_truth.shape' : (F.softmax(attention_scores_truth, dim=-1) * (attention_mask.transpose(-1, -2) > -1)).shape,
    'attention_probs_truth' : (F.softmax(attention_scores_truth, dim=-1) * (attention_mask.transpose(-1, -2) > -1)),

    'attention_probs_truth_m.shape' : (F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.transpose(-1, -2) > -1)).shape,
    'attention_probs_truth_m' : (F.softmax(resize_from_t_to_m(attention_scores_truth, T_M), dim=-1) * (attention_mask.transpose(-1, -2) > -1)),

    'col_select_mask.shape' : col_select_mask.shape if col_select_method == "sum_mask" else '',
    'col_select_mask' : col_select_mask if col_select_method == "sum_mask" else '',

    'sum_per_col.shape' : sum_per_col.shape,
    'sum_per_col' : sum_per_col,

    'largest_indx.shape' : largest_indx.shape,
    'largest_indx' : largest_indx,

    'large_inx_mask.shape' : large_inx_mask.shape,
    'large_inx_mask' : large_inx_mask,

    'col_sel_estimated_attention_probs_bef_select.shape' : col_sel_estimated_attention_probs1.shape,
    'col_sel_estimated_attention_probs_bef_select' : col_sel_estimated_attention_probs1,

    'col_sel_estimated_attention_probs_selcol_filled.shape' : col_sel_estimated_attention_probs.shape,
    'col_sel_estimated_attention_probs_selcol_filled' : col_sel_estimated_attention_probs,

    't_dead_mask.shape' : t_dead_mask.shape,
    't_dead_mask' : t_dead_mask,

    'partial_attention_mask_before_interp.shape' : partial_attention_mask1.shape,
    'partial_attention_mask_before_interp' : partial_attention_mask1,

    'partial_attention_mask_after_interp.shape' : partial_attention_mask.shape,
    'partial_attention_mask_after_interp' : partial_attention_mask,

    'attention_probs_dense.shape' : attention_probs_dense.shape,
    'attention_probs_dense' : attention_probs_dense,

    'partial_attention_probs.shape' : partial_attention_probs.shape,
    'partial_attention_probs' : partial_attention_probs,

    'partial_context_layer.shape' : partial_context_layer.shape,
    'partial_context_layer' : partial_context_layer,

    'estimated_attention_probs_for_output.shape' : estimated_attention_probs_for_output.shape,
    'estimated_attention_probs_for_output' : estimated_attention_probs_for_output
}

