import torch
from torch import nn

import ssm.third_party.openai_clip as clip
from ssm.third_party.openai_clip.simple_tokenizer import (
    SimpleTokenizer as _Tokenizer,
)


# Reuse one tokenizer for class-name length calculations.
_tokenizer = _Tokenizer()


# Encode learned prompt embeddings with the frozen CLIP text tower.
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        # CLIP stores the sentence representation at the end-of-text token.
        x = (
            x[
                torch.arange(x.shape[0]),
                tokenized_prompts.argmax(dim=-1),
            ]
            @ self.text_projection
        )
        return x


# Construct class prompts from trainable context and fixed CLIP tokens.
class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, args):
        super().__init__()
        n_cls = len(class_names)
        n_ctx = args.contexts_number
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Contexts may be class-specific or shared across all labels.
        if args.class_specific_contexts == "True":
            ctx_vectors = torch.empty(
                n_cls,
                n_ctx,
                ctx_dim,
                dtype=dtype,
            )
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        name_lens = [
            len(_tokenizer.encode(name))
            for name in class_names
        ]
        prompts = [
            prompt_prefix + " " + name
            for name in class_names
        ]

        tokenized_prompts = torch.cat([
            clip.tokenize(prompt)
            for prompt in prompts
        ])
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts
            ).type(dtype)
        # The start, class-name, and suffix embeddings remain fixed buffers.
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer(
            "token_suffix",
            embedding[:, 1 + n_ctx :, :],
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

    def forward(self):
        ctx = self.ctx
        # Broadcast a shared context to every class when required.
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # Assemble prompts according to the configured class-token position.
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[
                    i : i + 1,
                    :half_n_ctx,
                    :,
                ]
                ctx_i_half2 = ctx[
                    i : i + 1,
                    half_n_ctx:,
                    :,
                ]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts
