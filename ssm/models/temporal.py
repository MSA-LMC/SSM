import math

import torch
from einops import rearrange, repeat
from torch import einsum, nn


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Multi-head self-attention over the temporal token sequence.
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.heads
        # Split projected features into heads for scaled dot-product attention.
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b n (h d) -> b h n d",
                h=h,
            ),
            qkv,
        )
        dots = (
            einsum(
                "b h i d, b h j d -> b h i j",
                q,
                k,
            )
            * self.scale
        )
        attn = dots.softmax(dim=-1)
        out = einsum(
            "b h i j, b h j d -> b h i d",
            attn,
            v,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


# Stack pre-normalized attention and feed-forward residual blocks.
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                FeedForward(
                                    dim,
                                    mlp_dim,
                                    dropout=dropout,
                                ),
                            )
                        ),
                    ]
                )
            )

    def forward(self, x):
        # Preserve the original attention-then-FFN update order.
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# Summarize a frame sequence through a learned classification token.
class Temporal_Transformer_Cls(nn.Module):
    def __init__(
        self,
        num_patches,
        input_dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
    ):
        super().__init__()
        dropout = 0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, input_dim))
        self.temporal_transformer = Transformer(
            input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )
        self.ln = nn.LayerNorm(input_dim)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, n, _ = x.shape
        # Prepend the shared classification token to each clip.
        cls_tokens = repeat(
            self.cls_token,
            "() n d -> b n d",
            b=b,
        )
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        res = x
        x = self.temporal_transformer(x)
        cls_output = x[:, 0]
        # Gate the direct classification-token residual before normalization.
        return self.ln(cls_output + self.gate * res[:, 0])


class _LocalFrameAttention(nn.Module):
    """Apply a bounded, content-adaptive update from nearby frames."""

    def __init__(
        self,
        num_patches,
        input_dim,
        heads,
        dim_head,
        local_radius,
        max_update,
    ):
        super().__init__()
        if heads * dim_head != input_dim:
            raise ValueError(
                "Local AU attention requires heads * dim_head "
                f"== input_dim, got {heads} * {dim_head} "
                f"!= {input_dim}."
            )

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.max_update = float(max_update)
        self.norm = nn.LayerNorm(input_dim)
        self.to_q = nn.Linear(input_dim, input_dim, bias=False)
        self.to_k = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.xavier_uniform_(self.to_q.weight, gain=0.5)
        with torch.no_grad():
            self.to_k.weight.copy_(self.to_q.weight)

        # Prefer the current frame before the learned content scores mature.
        relative_bias = torch.zeros(heads, 2 * local_radius + 1)
        relative_bias[:, local_radius] = 1.0
        self.relative_position_bias = nn.Parameter(relative_bias)

        # ReZero-style per-channel mixing keeps the initial mapping exact.
        self.channel_gate = nn.Parameter(torch.zeros(input_dim))

        positions = torch.arange(num_patches)
        relative_offset = positions[None, :] - positions[:, None]
        local_mask = relative_offset.abs() <= local_radius
        relative_index = (
            relative_offset.clamp(-local_radius, local_radius) + local_radius
        )
        self.register_buffer(
            "local_attention_mask",
            local_mask.view(1, 1, num_patches, num_patches),
            persistent=False,
        )
        self.register_buffer(
            "relative_position_index",
            relative_index,
            persistent=False,
        )

    def forward(self, x):
        normalized = self.norm(x)
        q = rearrange(
            self.to_q(normalized),
            "b n (h d) -> b h n d",
            h=self.heads,
        )
        k = rearrange(
            self.to_k(normalized),
            "b n (h d) -> b h n d",
            h=self.heads,
        )
        scores = (
            einsum(
                "b h i d, b h j d -> b h i j",
                q,
                k,
            )
            * self.scale
        )
        relative_bias = self.relative_position_bias[:, self.relative_position_index]
        scores = scores + relative_bias.unsqueeze(0).to(dtype=scores.dtype)
        scores = scores.masked_fill(
            ~self.local_attention_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = scores.softmax(dim=-1)

        # Aggregate raw frame features so the update remains an interpolation
        # of real observations rather than a second arbitrary projection.
        values = rearrange(
            x,
            "b n (h d) -> b h n d",
            h=self.heads,
        )
        context = einsum(
            "b h i j, b h j d -> b h i d",
            weights,
            values,
        )
        context = rearrange(context, "b h n d -> b n (h d)")
        delta = context - x
        mixing = self.max_update * torch.tanh(self.channel_gate).view(1, 1, -1)
        mixing = mixing.to(dtype=x.dtype)
        return x + mixing * delta


# Learn short AU tracks without replacing the original frame representation.
class Temporal_Transformer_All(nn.Module):
    def __init__(
        self,
        num_patches,
        input_dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        local_radius=1,
        max_update=0.25,
    ):
        super().__init__()
        if num_patches <= 0:
            raise ValueError("num_patches must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if mlp_dim <= 0:
            raise ValueError("mlp_dim must be positive.")
        if not 0 < local_radius < num_patches:
            raise ValueError("local_radius must be in [1, num_patches - 1].")
        if not 0.0 < max_update <= 1.0:
            raise ValueError("max_update must be in (0, 1].")

        self.num_patches = num_patches
        self.input_dim = input_dim
        self.local_radius = local_radius
        self.layers = nn.ModuleList(
            [
                _LocalFrameAttention(
                    num_patches=num_patches,
                    input_dim=input_dim,
                    heads=heads,
                    dim_head=dim_head,
                    local_radius=local_radius,
                    max_update=max_update,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"AU temporal input must have shape [B, T, D], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_patches:
            raise ValueError(
                "AU temporal length mismatch: expected "
                f"{self.num_patches}, got {x.shape[1]}."
            )
        if x.shape[2] != self.input_dim:
            raise ValueError(
                "AU temporal feature mismatch: expected "
                f"{self.input_dim}, got {x.shape[2]}."
            )

        for layer in self.layers:
            x = layer(x)
        return x
