import warnings
from torch import nn
import torch
from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention

class BertBigBirdSelfAttention(BigBirdBlockSparseAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.max_seqlen = config.max_position_embeddings
        self.seed = None

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.bigbird_block_size = None
        self.bigbird_num_random_blocks = None

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        v : torch.Tensor,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        seed = None,
        block_size = 1,
        num_random_blocks = 2
    ):
        self.seed = seed
        self.bigbird_block_size = block_size
        self.bigbird_num_random_blocks = num_random_blocks

        # warnings.warn(f'seed {self.seed}')
        # NOTE JIN Currently this `class` can't be used in decoder.
        N, H, T, HID = q.shape

        to_seq_length = from_seq_length = T
        from_block_size = to_block_size = self.bigbird_block_size

        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        query_layer = q
        key_layer = k
        value_layer = v

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.bigbird_num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            N,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=True,
        )

        context_layer = context_layer.contiguous().view(N, from_seq_length, -1)
        
        outputs = (context_layer, attention_probs) # [N, T, H*HID], [N, H, T, T]
        return outputs
