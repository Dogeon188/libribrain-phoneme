"""
Meg2Vec: Multi-channel MEG adaptation of Wav2Vec2 architecture

This module provides a Meg2VecModel that adapts the proven Wav2Vec2 architecture
for multi-channel MEG (magnetoencephalography) data processing.

Key differences from Wav2Vec2:
- Input shape: (batch, channel_num, seq_len) instead of (batch, 1, seq_len)
- Multi-channel feature extraction: 306 MEG channels instead of single audio channel
- GroupNorm for spatial-temporal pattern learning across MEG sensors
- Designed for neuroscientific time-series data instead of audio

Architecture components:
1. Meg2VecFeatureExtractor: Multi-channel convolutional feature extraction
2. Meg2VecFeatureProjection: Projects conv features to transformer hidden dimension
3. Meg2VecPositionalConvEmbedding: Positional encoding via convolutions
4. Meg2VecEncoder: Multi-layer transformer encoder with self-attention

Usage:
    model = Meg2VecModel(
        sequence_length=500,
        in_channels=306,
        encoder_out_channels=256,
        hidden_size=768,
        num_hidden_layers=12
    )
    
    # Input: (batch_size, 306, 500) - MEG format
    input_tensor = torch.randn(batch_size, 306, 500)
    output = model(input_tensor)
    # Output: {"last_hidden_state": (batch_size, reduced_seq_len, hidden_size)}
"""

from torch import nn
from torch.nn import Conv1d, GELU, GroupNorm, Linear, LayerNorm, Dropout, TransformerEncoder, TransformerEncoderLayer
import torch
import math
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


class Meg2VecGroupNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels,
                           kernel_size, stride, bias=False)
        self.gelu = GELU()
        self.group_norm = GroupNorm(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.group_norm(x)
        return x


class Meg2VecConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels,
                           kernel_size, stride, bias=False)
        self.gelu = GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        return x


class Meg2VecFeatureExtractor(nn.Module):
    """Feature extraction from multi-channel MEG data similar to Wav2Vec2FeatureEncoder"""
    
    def __init__(
        self,
        sequence_length=500,
        in_channels=306,
        encoder_out_channels=256,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Meg2VecGroupNormConvLayer(
                in_channels, encoder_out_channels, 10, 5),
            Meg2VecConvLayer(
                encoder_out_channels, encoder_out_channels, 3, 2),
            Meg2VecConvLayer(
                encoder_out_channels, encoder_out_channels, 3, 2),
            Meg2VecConvLayer(
                encoder_out_channels, encoder_out_channels, 2, 2),
        )
        # Use LayerNorm that only normalizes over the channel dimension
        self.layer_norm = nn.LayerNorm(encoder_out_channels)

    def forward(self, x):
        x = self.conv_layers(x)
        # Transpose to (batch, seq, channels) for LayerNorm, then back
        x = x.transpose(1, 2)  # (batch, seq, channels)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, channels, seq)
        return x


class Meg2VecFeatureProjection(nn.Module):
    """Feature projection layer similar to Wav2Vec2FeatureProjection"""
    
    def __init__(self, conv_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(conv_dim, elementwise_affine=True)
        self.projection = Linear(conv_dim, hidden_size)
        self.dropout = Dropout(dropout)

    def forward(self, hidden_states):
        # normalize input features
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Meg2VecPositionalConvEmbedding(nn.Module):
    """Positional convolutional embedding similar to Wav2Vec2PositionalConvEmbedding"""
    
    def __init__(self, hidden_size, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16):
        super().__init__()
        self.conv = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
        )
        # Apply weight normalization to conv layer
        weight_norm = nn.utils.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)
        self.padding = Meg2VecSamePadLayer(num_conv_pos_embeddings)
        self.activation = GELU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)  # (batch, time, hidden) -> (batch, hidden, time)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (batch, hidden, time) -> (batch, time, hidden)
        return hidden_states


class Meg2VecSamePadLayer(nn.Module):
    """Same padding layer for conv layers"""
    
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states


class Meg2VecEncoder(nn.Module):
    """Transformer encoder similar to Wav2Vec2Encoder"""
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.pos_conv_embed = Meg2VecPositionalConvEmbedding(hidden_size)
        self.layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = Dropout(hidden_dropout)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=attention_dropout,
            activation='gelu',
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.layers = TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Add positional embeddings
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        
        # Apply layer normalization and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Pass through transformer layers
        # Note: PyTorch's TransformerEncoder expects (batch, seq, hidden) when batch_first=True
        encoded_states = self.layers(hidden_states, src_key_padding_mask=attention_mask)
        
        if return_dict:
            return {
                'last_hidden_state': encoded_states,
                'hidden_states': (hidden_states, encoded_states) if output_hidden_states else None,
            }
        return encoded_states


class Meg2VecModel(nn.Module):
    """
    Meg2Vec model adapted from Wav2Vec2 for multi-channel MEG data.
    
    Input shape: (batch_size, channel_num, seq_len) where channel_num=306 for MEG
    Output shape: (batch_size, reduced_seq_len, hidden_size)
    """

    def __init__(
        self,
        sequence_length=500,
        in_channels=306,
        encoder_out_channels=256,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        feature_projection_dropout=0.1,
    ):
        super().__init__()
        
        # Feature extraction - extract features from multi-channel input
        self.feature_extractor = Meg2VecFeatureExtractor(
            sequence_length=sequence_length,
            in_channels=in_channels,
            encoder_out_channels=encoder_out_channels
        )
        
        # Calculate feature dimensions after conv layers
        dummy_input = torch.zeros(1, in_channels, sequence_length)
        with torch.no_grad():
            feature_output = self.feature_extractor(dummy_input)
            conv_dim = feature_output.shape[1]  # encoder_out_channels
            self.sequence_length_after_conv = feature_output.shape[2]
        
        # Feature projection - project to hidden dimension
        self.feature_projection = Meg2VecFeatureProjection(
            conv_dim=conv_dim,
            hidden_size=hidden_size,
            dropout=feature_projection_dropout
        )
        
        # Transformer encoder
        self.encoder = Meg2VecEncoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
        )

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Forward pass through Meg2Vec model.
        
        Args:
            input_values: Tensor of shape (batch_size, channel_num, seq_len)
            attention_mask: Optional attention mask
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Dict or tuple containing:
                - last_hidden_state: (batch_size, reduced_seq_len, hidden_size)
                - hidden_states: Optional tuple of hidden states
        """
        # Extract features: (batch, channel, seq) -> (batch, conv_dim, reduced_seq)
        extract_features = self.feature_extractor(input_values)
        
        # Transpose for transformer: (batch, conv_dim, reduced_seq) -> (batch, reduced_seq, conv_dim)
        extract_features = extract_features.transpose(1, 2)
        
        # Project to hidden dimension: (batch, reduced_seq, conv_dim) -> (batch, reduced_seq, hidden_size)
        hidden_states = self.feature_projection(extract_features)
        
        # Create attention mask if not provided
        if attention_mask is not None:
            # Downsample attention mask to match reduced sequence length
            # This is a simplified approach - in practice you might want more sophisticated mask handling
            batch_size = attention_mask.shape[0]
            attention_mask = attention_mask[:, :self.sequence_length_after_conv]
            if attention_mask.shape[1] < self.sequence_length_after_conv:
                # Pad if needed
                padding = self.sequence_length_after_conv - attention_mask.shape[1]
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.zeros(batch_size, padding, dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=1)
            # Convert to mask format expected by transformer (True for positions to mask)
            attention_mask = attention_mask == 0
        
        # Pass through transformer encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if return_dict:
            return encoder_outputs
        else:
            return encoder_outputs


@dataclass
class Meg2VecForPreTrainingOutput:
    """
    Output type of [`Meg2VecForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d).
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, proj_codevector_dim)`):
            Hidden-states of the model projected to proj_codevector_dim that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, proj_codevector_dim)`):
            Quantized extracted feature vectors projected to proj_codevector_dim representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m).
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d).
    """

    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement SpecAugment masking.
    
    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob: The percentage of the whole axis (between 0 and 1) which will be masked.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)
        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length
        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)
        return num_masked_span

    # compute number of masked spans in batch
    if attention_mask is not None:
        if hasattr(attention_mask, 'detach'):
            # It's a tensor
            input_lengths = attention_mask.sum(-1).detach().tolist()
        else:
            # It's a numpy array
            input_lengths = attention_mask.sum(-1).tolist()
    else:
        input_lengths = [sequence_length for _ in range(batch_size)]

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this sample
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        if num_masked_span == 0:
            spec_aug_mask_idxs.append([])
        else:
            spec_aug_mask_idx = np.random.choice(
                np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
            )
            spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # Pad to max_num_masked_span
    for i, spec_aug_mask_idx in enumerate(spec_aug_mask_idxs):
        if len(spec_aug_mask_idx) < max_num_masked_span:
            spec_aug_mask_idxs[i] = np.concatenate(
                [spec_aug_mask_idx, np.ones(max_num_masked_span - len(spec_aug_mask_idx), dtype=np.int32) * -1]
            )

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        num_masked = mask_time_indices[batch_idx].sum()
        
        if num_masked == 0:
            # No masked positions, sample randomly from all positions
            sampled_negative_indices[batch_idx] = np.random.choice(
                sequence_length, size=(sequence_length, num_negatives), replace=True
            )
            continue
            
        # Get indices of masked positions
        masked_positions = np.where(mask_time_indices[batch_idx])[0]
        
        # For each masked position, sample negative indices
        for pos_idx, masked_pos in enumerate(masked_positions):
            # Sample from all available positions
            available_positions = np.arange(sequence_length)
            
            # Remove the current masked position to avoid self-sampling
            available_positions = available_positions[available_positions != masked_pos]
            
            if len(available_positions) == 0:
                # Edge case: only one position total, sample with replacement
                negative_samples = np.full(num_negatives, 0, dtype=np.int32)
            elif len(available_positions) < num_negatives:
                # Sample with replacement if not enough unique positions
                negative_samples = np.random.choice(
                    available_positions, size=num_negatives, replace=True
                )
            else:
                # Sample without replacement
                negative_samples = np.random.choice(
                    available_positions, size=num_negatives, replace=False
                )
            
            sampled_negative_indices[batch_idx, masked_pos] = negative_samples

    return sampled_negative_indices


class Meg2VecGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax for MEG data.
    """

    def __init__(
        self,
        num_codevector_groups=2,
        num_codevectors_per_group=320,
        codevector_dim=256,
        conv_dim=256,
    ):
        super().__init__()
        self.num_groups = num_codevector_groups
        self.num_vars = num_codevectors_per_group

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`codevector_dim {codevector_dim} must be divisible "
                f"by `num_codevector_groups` {self.num_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(conv_dim, self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # compute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity


class Meg2VecForPreTraining(nn.Module):
    """
    Meg2Vec model for self-supervised pre-training using contrastive learning.
    
    Similar to Wav2Vec2ForPreTraining but adapted for multi-channel MEG data.
    """

    def __init__(
        self,
        sequence_length=500,
        in_channels=306,
        encoder_out_channels=256,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        feature_projection_dropout=0.1,
        # Quantization parameters
        num_codevector_groups=2,
        num_codevectors_per_group=320,
        codevector_dim=256,
        proj_codevector_dim=256,
        feat_quantizer_dropout=0.1,
        # Contrastive learning parameters
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        diversity_loss_weight=0.1,
    ):
        super().__init__()
        
        # Base model
        self.meg2vec = Meg2VecModel(
            sequence_length=sequence_length,
            in_channels=in_channels,
            encoder_out_channels=encoder_out_channels,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            feature_projection_dropout=feature_projection_dropout,
        )
        
        # Quantization
        self.dropout_features = nn.Dropout(feat_quantizer_dropout)
        self.quantizer = Meg2VecGumbelVectorQuantizer(
            num_codevector_groups=num_codevector_groups,
            num_codevectors_per_group=num_codevectors_per_group,
            codevector_dim=codevector_dim,
            conv_dim=encoder_out_channels,
        )
        
        # Projection layers
        self.project_hid = nn.Linear(hidden_size, proj_codevector_dim)
        self.project_q = nn.Linear(codevector_dim, proj_codevector_dim)
        
        # Configuration
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.num_negatives = num_negatives
        self.diversity_loss_weight = diversity_loss_weight
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        for param in self.meg2vec.feature_extractor.parameters():
            param.requires_grad = False

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        # Simulate the conv layers from Meg2VecFeatureExtractor
        for kernel_size, stride in [(10, 5), (3, 2), (3, 2), (2, 2)]:
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask, add_adapter=None):
        """
        Compute reduced attention mask corresponding to feature vectors
        """
        # Non-padded positions are marked with 1
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # Get corresponding lengths after conv layers
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: float = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits

    def forward(
        self,
        input_values,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Meg2VecForPreTrainingOutput]:
        """
        Args:
            input_values: Tensor of shape (batch_size, channel_num, seq_len)
            attention_mask: Optional attention mask
            mask_time_indices: Indices to mask extracted features for contrastive loss
            sampled_negative_indices: Indices indicating which quantized target vectors are used as negative sampled vectors
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple

        Returns:
            Meg2VecForPreTrainingOutput or tuple containing:
                - loss: Total loss (contrastive + diversity)
                - projected_states: Projected transformer features
                - projected_quantized_states: Projected quantized features
                - codevector_perplexity: Perplexity of the quantized vectors
        """
        return_dict = return_dict if return_dict is not None else True

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        # Forward through base model
        outputs = self.meg2vec(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Get transformer features and extracted features
        transformer_features = outputs['last_hidden_state']  # (batch, seq, hidden_size)
        
        # Get extract features - we need to get them from feature extractor
        # Extract features: (batch, channel, seq) -> (batch, conv_dim, reduced_seq)
        extract_features = self.meg2vec.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)  # (batch, reduced_seq, conv_dim)

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(transformer_features)

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(extract_features)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )

        quantized_features = quantized_features.to(self.project_q.weight.dtype)
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.num_codevectors_per_group * self.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + (outputs.get('hidden_states'), outputs.get('attentions'))
            return (transformer_features, quantized_features, codevector_perplexity) + (outputs.get('hidden_states'), outputs.get('attentions'))

        return Meg2VecForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.get('hidden_states'),
            attentions=outputs.get('attentions'),
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )
