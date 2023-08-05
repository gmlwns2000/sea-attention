
# batch]
# mean_values
# sum_mask
# TODO check per_item_col_thickness for self.benchmarking


!! col_sel_estimated_attention_probs is changed to t !!
# col_veiw and t shape is different!!

// get t based on k methods # NOTE memory order differs with topk
col_view_estimated_attention_probs = masked_estimated_attention_probs.permute(0, 2, 1, 3).view(N, T, H*T_M) if batch else masked_estimated_attention_probs
t = masked_estimated_attention_probs.permute(0, 2, 1, 3).view(N, T*H*T_M) if batch else masked_estimated_attention_probs.view(N, H, T*T_M)

// get per_item_col_thickness
per_item_col_thickness = min(max(1,round(T_M/token_length)),T_M) # N, 1
max_col_thickness = int(max(per_item_col_thickness))

// col_mean_in_t # t shape (N, H*T_M)
col_mean_in_flatten = col_view_estimated_attention_probs.sum(dim=-2)/token_length # N, col_view_estimated_attention_probs.shape[-1]

// col_result_mask = col_mean_in_t >= (1/T_M)  # (N, H*T_M) # NOTE hard coded value
// col_result_mask_cnt = col_result_mask.sum(dim=-1, keepdim=True) # N, 1

# need to pick indices based on col_mean_in_t (sort)
// get indices_col
// inflated_per_item_top_k_col = (min(col_result_mask_cnt, per_item_top_k))*per_item_col_thickness
// inflated_top_k_elems_col = max(inflated_per_item_top_k_col)
_, indices = torch.topk(
    input=col_mean_in_flatten, # N, H*T_M
    k=inflated_top_k_elems_col, 
    dim=-1, 
    sorted=True #sorted true is important
) # N, inflated_top_k_elems_col

# partial_attention_mask_col aranged with inflated_top_k_elems_col
// partial_attention_mask_col = t.shape, fill with t.shape[-1]

# get 
// partial_attention_mask_col.scatter_(
    dim=-1,
    index=indices,
    src=torch.arange(
        inflated_top_k_elems_col,
        dtype=torch.long,
        device=attention_mask.device,
    )
)

// t_alive_mask_col = partial_attention_mask_col < inflated_per_item_top_k_col
// t_alive_mask_col_t = m_to_t(t_dead_mask_col)
// t_alive_mask_col_m = t_to_m(t_dead_mask_col) # N, T*H*T_M

// get col_mask_indices of 1 in col_t_alive_mask_flatten
# col_mask_indices = torch.nonzero(t_alive_mask_col_m) # N, T*H*T_M <-not right code

// fill -inf in score, 0 in probs in that indices
// permute+view score, probs before topk

// per_item_top_k_col_real = t_alive_mask_col_m.sum(dim=-1, keepdim=True)
assert it's smaller than per_item_top_k
// update per_item_top_k (HJ) <- change t to the right shape tensor


// HJ topk 

// t_dead_mask in topk (HJ) [N, H*T*T_M]
// t_dead_mask [N, H*T*T_M] -> [N, H, T, T_M] -> [N, T, H, T_M] -> [N, T*H*T_M]
// t_dead_mask.fill(value=1, indices=col_mask_indices)




# per_item_col_result_width = H*(T_M-per_item_col_thickness+1) if batch else T_M-per_item_col_thickness+1 # N,1
# max_col_result_width = int(max(per_item_col_result_width)) 

col_result = torch.zeros(N, max_col_result_width) # N,1


if batch:
    total_km_upperbound_2d = H*k_m/per_item_col_thickness *token_length
elif head:
    total_km_upperbound_2d = k_m/per_item_col_thickness *token_length



if mean_values:
    // get mean values per column
    # -2 dim in t must have preconsidered padding(masked_attention_probs<-attention_mask)
    col_mean_in_flatten = t.sum(dim=-2)/token_length # N, t.shape[-1]

    // mask values higher than 1/T_M
    col_result_mask = col_mean_in_flatten >= 1/T_M

    // select top values that satisfies col_result_mask : selected cnt should be 0~H*k_m*(1/col_thickness)
    // 
    // the selected ones should be in the same things #NOTE should select nearby columns
    topk_col = k_m*per_item_col_thickness
    // 


    


# head]
# mean_values
# sum_mask



