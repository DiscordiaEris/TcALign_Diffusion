"""Keep FaceID's query and output projections independent of text fine-tuning."""

from copy import deepcopy
from numbers import Integral, Real

import torch
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    IPAdapterAttnProcessor2_0,
)
from torch import nn
from torch.nn import functional as F


class FrozenIdentityAttnProcessor(nn.Module):
    """A single SD1.5 FaceID branch with frozen private Q/output projections.

    Construct after installing FaceID's LoRA hooks and before updating the text
    attention: ``attn.set_processor(FrozenIdentityAttnProcessor(attn, attn.processor))``.
    Copies include the companion LoRA buffers/hooks. The text branch retains the
    live attention module; the identity branch keeps gradients to its input.
    An optional low-rank output residual learns from raw identity attention,
    with a zero-initialized up projection preserving the starting output.

    Supports BNC/BCHW inputs and one identity image, with the diffusers tuple
    ``(text_tokens, [image_tokens])``. Identity masks, multiple adapters/images,
    nonzero output dropout, and non-SD1.5 Q/K normalization are rejected.
    """

    def __init__(self, attn, processor, *, train_identity_gain=False,
                 identity_refinement_rank=0):
        super().__init__()
        if (isinstance(identity_refinement_rank, bool)
                or not isinstance(identity_refinement_rank, Integral)
                or identity_refinement_rank < 0):
            raise ValueError("Identity refinement rank must be a nonnegative integer")
        if not isinstance(processor, IPAdapterAttnProcessor2_0):
            raise TypeError("Expected an IPAdapterAttnProcessor2_0")
        if len(processor.to_k_ip) != 1 or len(processor.to_v_ip) != 1:
            raise ValueError("Only a single identity adapter is supported")
        if len(processor.num_tokens) != 1 or len(processor.scale) != 1:
            raise ValueError("Only a single identity adapter scale is supported")
        if not isinstance(attn.to_out[1], nn.Dropout) or attn.to_out[1].p != 0:
            raise NotImplementedError("Identity isolation requires zero output dropout")
        if attn.norm_q is not None or attn.norm_k is not None:
            raise NotImplementedError(
                "Only SD1.5 attention without Q/K normalization is supported"
            )
        self.hidden_size = processor.hidden_size
        self.cross_attention_dim = processor.cross_attention_dim
        self.num_tokens = list(processor.num_tokens)
        self.scale = list(processor.scale)
        self.to_k_ip = processor.to_k_ip
        self.to_v_ip = processor.to_v_ip
        self.to_q_identity = deepcopy(attn.to_q)
        self.to_out_identity = deepcopy(attn.to_out[0])
        # The text baseline already adds this affine bias once. The identity
        # residual must contain only W_out @ identity_attention (+ its LoRA).
        self.to_out_identity.bias = None
        self.group_norm_identity = deepcopy(attn.group_norm)
        self.spatial_norm_identity = deepcopy(attn.spatial_norm)
        self.text_processor = AttnProcessor2_0()
        self.requires_grad_(False)
        if identity_refinement_rank > 0:
            self.identity_refinement = nn.Sequential(
                nn.Linear(self.hidden_size, identity_refinement_rank, bias=False),
                nn.Linear(identity_refinement_rank, self.hidden_size, bias=False),
            ).to(attn.to_q.weight)
            nn.init.zeros_(self.identity_refinement[1].weight)
        if train_identity_gain:
            self.identity_gain_logit = nn.Parameter(attn.to_q.weight.new_zeros(()))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        ip_adapter_masks=None,
    ):
        if ip_adapter_masks is not None:
            raise NotImplementedError("Identity attention masks are not supported")
        if (
            not isinstance(encoder_hidden_states, tuple)
            or len(encoder_hidden_states) != 2
        ):
            raise ValueError("Expected the tuple (text_tokens, [image_tokens])")
        text_tokens, image_tokens = encoder_hidden_states
        if not isinstance(image_tokens, (list, tuple)) or len(image_tokens) != 1:
            raise ValueError("Expected a single identity adapter")
        image_tokens = image_tokens[0]
        if image_tokens.ndim == 4:
            if image_tokens.shape[1] != 1:
                raise ValueError("Only one identity image is supported")
            image_tokens = image_tokens[:, 0]
        if image_tokens.ndim != 3:
            raise ValueError("Identity tokens must have shape BNC or B1NC")
        if hidden_states.ndim not in (3, 4):
            raise ValueError("Attention hidden states must have shape BNC or BCHW")
        if text_tokens.ndim != 3 or not (
            len(text_tokens) == len(hidden_states) == len(image_tokens)
        ):
            raise ValueError(
                "Text, hidden states, and identity tokens must have matching batches"
            )
        if not isinstance(self.scale, (list, tuple)) or len(self.scale) != 1:
            raise ValueError("Expected one numeric identity scale")
        identity_scale = self.scale[0]
        if not isinstance(identity_scale, Real):
            raise ValueError("Expected one numeric identity scale")

        text_output = self.text_processor(
            attn,
            hidden_states,
            text_tokens,
            attention_mask=attention_mask,
            temb=temb,
        )
        if identity_scale == 0:
            return text_output
        if hasattr(self, "identity_gain_logit"):
            identity_scale = identity_scale * (0.5 + self.identity_gain_logit.sigmoid())

        # Match preprocessing before the original shared query. Private copies
        # also keep optional normalizations frozen if the text side is trained.
        identity_hidden = hidden_states
        if self.spatial_norm_identity is not None:
            identity_hidden = self.spatial_norm_identity(identity_hidden, temb)
        if hidden_states.ndim == 4:
            batch, channel, height, width = hidden_states.shape
            identity_hidden = identity_hidden.reshape(
                batch, channel, height * width
            ).transpose(1, 2)
        else:
            batch = len(hidden_states)
        if self.group_norm_identity is not None:
            identity_hidden = self.group_norm_identity(
                identity_hidden.transpose(1, 2)
            ).transpose(1, 2)

        query = self.to_q_identity(identity_hidden)
        key = self.to_k_ip[0](image_tokens)
        value = self.to_v_ip[0](image_tokens)
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        identity = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        raw_identity = (
            identity.transpose(1, 2)
            .reshape(batch, -1, attn.heads * head_dim)
            .to(query.dtype)
        )
        identity = self.to_out_identity(raw_identity)
        if hasattr(self, "identity_refinement"):
            identity = identity + self.identity_refinement(raw_identity)
        if hidden_states.ndim == 4:
            identity = identity.transpose(1, 2).reshape(batch, channel, height, width)
        # Residual and affine bias were already included by the text processor.
        return text_output + identity_scale * identity / attn.rescale_output_factor
