# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sinkhorn Attention Transformer models."""
from flax.deprecated import nn
import jax.numpy as jnp
from _lra_benchmarks.models.layers import common_layers
from _lra_benchmarks.models.sinkhorn_transformer import sinkhorn_attention


class SinkhornTransformerBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None,
            block_size=50):
    """Applies TransformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.
      block_size: int, block size.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = sinkhorn_attention.SinkhornSelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        block_size=block_size)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class SinkhornTransformerEncoder(nn.Module):
  """Sinkhorn Transformer Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            block_size=50,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='MEAN',
            num_classes=10):
    """Applies Local Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      block_size: int, block size.
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(num_layers):
      x = SinkhornTransformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}',
          block_size=block_size)
    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      if classifier_pool == 'MEAN':
        encoded = jnp.mean(encoded, axis=1)
        encoded = nn.Dense(encoded, num_classes, name='logits')
      else:
        # TODO(yitay): Add other pooling methods.
        raise ValueError('Pooling method not supported yet.')
    return encoded


class LocalTransformerDecoder(nn.Module):
  """Local Transformer Decoder."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            cache=None,
            block_size=50):
    """Applies Local Attention model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: bool: if model is training.
      shift: bool: if we right-shift input - this is only disabled for
        fast, looped single-token autoregressive decoding.
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      cache: flax autoregressive cache for fast decoding.
      block_size: int, block size

    Returns:
      output of a transformer decoder.
    """
    padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[..., None]
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if shift:
      x = common_layers.shift_right(x)
    x = x.astype('int32')
    x = common_layers.Embed(
        x, num_embeddings=vocab_size, features=emb_dim, name='embed')
    x = common_layers.AddPositionEmbs(
        x,
        max_len=max_len,
        posemb_init=common_layers.sinusoidal_init(max_len=max_len),
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    for _ in range(num_layers):
      x = SinkhornTransformerBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=True,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          cache=cache,
          block_size=block_size)
    x = nn.LayerNorm(x)
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits


class SinkhornTransformerDualEncoder(nn.Module):
  """Sinkhorn Transformer Model for Matching (dual encoding) tasks."""

  def apply(self,
            inputs1,
            inputs2,
            vocab_size=None,
            inputs1_positions=None,
            inputs2_positions=None,
            inputs1_segmentation=None,
            inputs2_segmentation=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            classifier=True,
            classifier_pool='CLS',
            num_classes=2,
            interaction=None):
    """Applies Transformer model on text similarity.

    A deliberate choice to distinguish this from NLI because
    we may want to do different things to the model later. Dual Encoding
    mode enforces that we do not do cross attention between pairs.

    Args:
      inputs1: input data.
      inputs2: target data.
      vocab_size: size of the input vocabulary.
      inputs1_positions: input subsequence positions for packed examples.
      inputs2_positions: target subsequence positions for packed examples.
      inputs1_segmentation: input segmentation info for packed examples.
      inputs2_segmentation: target segmentation info for packed examples.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      classifier: boolean, to use classifier.
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      interaction: str.

    Returns:
      output of a transformer decoder.
    """
    encoder = SinkhornTransformerEncoder.shared(
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation,
        vocab_size=vocab_size,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')
    inputs1_encoded = encoder(inputs1)
    inputs2_encoded = encoder(inputs2)

    encoded = common_layers.classifier_head_dual(
        inputs1_encoded,
        inputs2_encoded,
        num_classes,
        mlp_dim,
        pooling_mode=classifier_pool,
        interaction=interaction)

    return encoded
