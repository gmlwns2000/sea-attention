import math
from torch import nn
import torch
from typing import Tuple

from transformers import LongformerSelfAttention

# NOTE global attention is based on [CLS]
# TODO need to change 2w+1 to 2w+2 or add row global pattern
# TODO is_index_global_attn shape is [N, T], check line 237
# TODO check hidden states padding in perlin_bert
# TODO check attention mechanism: diagonal calculated twice? but the size is 2w+1 tho

class BertLongformerSelfAttention(LongformerSelfAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.config = config
        self.one_sided_attn_window_size = 2
    
    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        N, T, num_heads, head_dim = query.size()
        assert (
            T % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {T}"
        assert query.size() == key.size()
        chunks_count = torch.div(T, window_overlap, rounding_mode="trunc") - 1
        # group N and num_heads dimensions into one, then chunk T into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(N * num_heads, T, head_dim)
        key = key.transpose(1, 2).reshape(N * num_heads, T, head_dim)
        query = self._chunk(query, window_overlap, getattr(self.config, "onnx_export", False))
        key = self._chunk(key, window_overlap, getattr(self.config, "onnx_export", False))
        # matrix multiplication
        # bcxd: N * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: N * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: N * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (N * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )
        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ] # [N*H, chunk_count, w, 2w+1]

        # separate N and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            N, num_heads, T, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores.float(), window_overlap) # NOTE added .float()
        return diagonal_attention_scores
        
    def forward(
        self,
        hidden_states, # [N, T, H*HID]
        q : torch.Tensor, # [N, H, T, HID]
        k : torch.Tensor, 
        v : torch.Tensor,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        one_sided_attn_window_size = 2,
        attention_mask=None,
        layer_head_mask=None,
        )-> Tuple[torch.Tensor, torch.Tensor]: 
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1) # [N, T, H*HID]->[T, N, H*HID]

        T, N, embed_dim = hidden_states.size() # [T, N, H*HID]
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors = q.permute(2, 0, 1, 3).contiguous().view(T, N, embed_dim) # [N, H, T, HID]->[T, N, H, HID]->[T, N, H*HID]
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(T, N, self.num_heads, self.head_dim).transpose(0, 1) # [N, T, H, HID]
        key_vectors = k.transpose(1, 2) # [N, H, T, HID]->[N, T, H, HID]
        value_vectors = v.transpose(1,2)

        self.one_sided_attn_window_size = one_sided_attn_window_size

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, one_sided_attn_window_size
        ) # [N, T, H, 2w+1]
        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, one_sided_attn_window_size
        ) # [N, T, 1, 2w+1]
        # pad local attention probs
        attn_scores += diagonal_mask
        assert list(attn_scores.size()) == [
            N,
            T,
            self.num_heads,
            one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({N}, {T}, {self.num_heads},"
            f" {one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )
        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (N, T, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            # free memory
            del global_key_attn_scores
        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)
        # free memory
        del attn_scores
        
        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, one_sided_attn_window_size
            )
        assert attn_output.size() == (N, T, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(T, N, embed_dim).contiguous() # [T, N, H*HID]
        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )
            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]
            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0 # [16, 204, 12, 6] NOTE 0 is_index_global_attn_nonzero (0,0), (1,0) to (16, 0) might be a bug
        outputs = (attn_output.transpose(0, 1),) # [T, N, H*HID]->[N, T, H*HID]
        attn_probs = attn_probs.transpose(1, 2) # [N, T, H, 6<2w+1+global(1)>]->[N, H, T, 2w+1+global(1)]
        global_attn_probs = global_attn_probs.transpose(2, 3) # [N, H, 1, T]->[N, H, T, 1]
        attn_probs[:, :, :, 0] = global_attn_probs.squeeze() # NOTE put global_attn_probs to attn_probs
        outputs += (attn_probs,) 
        return outputs