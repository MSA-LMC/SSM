import math

import torch
from einops import rearrange, repeat
from torch import einsum, nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1
            + torch.tanh(
                math.sqrt(2 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3))
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
        self.scale = dim_head ** -0.5
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
        dots = einsum(
            "b h i d, b h j d -> b h i j",
            q,
            k,
        ) * self.scale
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
                nn.ModuleList([
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
                ])
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
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, input_dim)
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, input_dim)
        )
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
        return self.ln(
            cls_output + self.gate * res[:, 0]
        )


# Contextualize every frame while preserving the full temporal sequence.
class Temporal_Transformer_All(nn.Module):
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
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, input_dim)
        )
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
        # Restore the configured clip structure before temporal attention.
        x = x.contiguous().view(
            -1,
            self.num_patches,
            self.input_dim,
        )
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        res = x
        x = self.temporal_transformer(x)
        # Return framewise features with a gated residual connection.
        return self.ln(x + self.gate * res)
