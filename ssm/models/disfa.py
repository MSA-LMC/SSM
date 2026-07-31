import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import patch_visual_moe
from .prompting import PromptLearner, TextEncoder
from .temporal import (
    Temporal_Transformer_All,
    Temporal_Transformer_Cls,
)


class DisfaSSM(nn.Module):
    """Legacy-compatible SSM model for the eight-AU DISFA setting."""

    def __init__(
        self,
        input_text1,
        input_text2,
        input_text3,
        input_text4,
        clip_model,
        args,
    ):
        super().__init__()
        emotion_count = len(input_text1)
        if emotion_count not in {7, 11}:
            raise ValueError(
                f"Unsupported emotion class count: {emotion_count}"
            )
        if len(input_text2) != 8:
            raise ValueError(
                "DISFA requires eight action-unit descriptions."
            )
        self.args = args
        self.dtype = clip_model.dtype

        # Learn separate CLIP contexts for emotion and AU descriptions.
        self.input_text1 = input_text1
        self.prompt_learner1 = PromptLearner(
            input_text1,
            clip_model,
            args,
        )
        self.tokenized_prompts1 = (
            self.prompt_learner1.tokenized_prompts
        )

        self.input_text2 = input_text2
        self.prompt_learner2 = PromptLearner(
            input_text2,
            clip_model,
            args,
        )
        self.tokenized_prompts2 = (
            self.prompt_learner2.tokenized_prompts
        )

        self.text_encoder = TextEncoder(clip_model)

        # Replace the final visual MLP blocks with shared/private MoE FFNs.
        self.encoder = patch_visual_moe(
            clip_model,
            n_private_experts=4,
            expert_hidden_dim=512,
            noise_std=1e-2,
        )

        # Pool expression clips globally while retaining framewise AU features.
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=1,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
        )
        self.temporal_net_au = Temporal_Transformer_All(
            num_patches=16,
            input_dim=512,
            depth=1,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
        )
        self.au_head = nn.Linear(512, 8)
        self.dfer_head = nn.Linear(512, emotion_count)

        # Initialize the fixed depthwise averaging filter used for AU logits.
        K = getattr(args, "smooth_K", 2)
        window = 2 * K + 1
        self.au_smooth = nn.Conv1d(
            in_channels=8,
            out_channels=8,
            kernel_size=window,
            padding=K,
            groups=8,
            bias=False,
        )
        nn.init.constant_(
            self.au_smooth.weight,
            1.0 / window,
        )

        # Seed cross-task transfer with the fixed emotion-to-AU prior.
        init_map = torch.tensor(
            [
                [
                    0.02,
                    0.02,
                    0.02,
                    1.00,
                    0.02,
                    1.00,
                    0.02,
                    0.02,
                ],
                [
                    1.00,
                    0.02,
                    1.00,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                ],
                [
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                ],
                [
                    0.02,
                    0.02,
                    1.00,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                ],
                [
                    1.00,
                    1.00,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    1.00,
                    1.00,
                ],
                [
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    1.00,
                    0.02,
                    0.02,
                    0.02,
                ],
                [
                    1.00,
                    1.00,
                    1.00,
                    0.02,
                    0.02,
                    0.02,
                    1.00,
                    1.00,
                ],
            ],
            dtype=torch.float32,
        )
        # Use the extended prior when the 11 MAFW descriptions are supplied.
        if emotion_count == 11:
            init_map = torch.tensor(
                [
                    [0.02, 0.02, 0.02, 1.00, 0.02, 1.00, 0.02, 0.02],
                    [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02],
                    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                    [0.02, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02],
                    [1.00, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00],
                    [0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02],
                    [1.00, 1.00, 1.00, 0.02, 0.02, 0.02, 0.02, 1.00],
                    [0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02],
                    [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 1.00, 0.02],
                    [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 1.00],
                    [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 1.00, 0.02],
                ],
                dtype=torch.float32,
            )
        self.register_buffer(
            "init_map",
            init_map,
        )

        row_targets_au2emo = init_map.sum(dim=1)
        row_targets_emo2au = init_map.sum(dim=0)

        self.register_buffer(
            "row_targets_au2emo",
            row_targets_au2emo,
        )
        self.register_buffer(
            "row_targets_emo2au",
            row_targets_emo2au,
        )

        # Learn residual map corrections and gated semantic fusion strengths.
        self.delta_map1 = nn.Parameter(
            torch.zeros_like(init_map)
        )
        self.delta_map2 = nn.Parameter(
            torch.zeros_like(init_map.t())
        )

        gate_init = math.log(0.1 / 0.9)
        self.alpha = nn.Parameter(
            torch.tensor(gate_init)
        )
        self.beta = nn.Parameter(
            torch.tensor(gate_init)
        )

    def _compute_maps(self, eps=1e-8):
        neutral_idx = 2

        # Preserve each prior row mass while learning AU contributions.
        logits1 = (
            torch.log(self.init_map + eps)
            + self.delta_map1
        )
        map_au2emo = F.softmax(logits1, dim=1)
        map_au2emo = (
            map_au2emo
            * self.row_targets_au2emo.view(-1, 1)
        )
        map_au2emo = torch.cat(
            [
                map_au2emo[:neutral_idx],
                self.init_map[
                    neutral_idx:neutral_idx + 1
                ],
                map_au2emo[neutral_idx + 1:],
            ],
            dim=0,
        )

        init_map_t = self.init_map.t()
        # Keep the neutral relation fixed in both mapping directions.
        non_neutral_mask = torch.ones(
            init_map_t.size(1),
            dtype=torch.bool,
            device=init_map_t.device,
        )
        non_neutral_mask[neutral_idx] = False

        logits2_non_neutral = (
            torch.log(
                init_map_t[:, non_neutral_mask] + eps
            )
            + self.delta_map2[:, non_neutral_mask]
        )
        map_emo2au_non_neutral = F.softmax(
            logits2_non_neutral,
            dim=1,
        )

        neutral_column = init_map_t[
            :,
            neutral_idx:neutral_idx + 1,
        ]
        remaining_targets = (
            self.row_targets_emo2au.view(-1, 1)
            - neutral_column
        )
        map_emo2au_non_neutral = (
            map_emo2au_non_neutral
            * remaining_targets
        )

        map_emo2au = torch.zeros_like(init_map_t)
        map_emo2au[:, non_neutral_mask] = (
            map_emo2au_non_neutral
        )
        map_emo2au[
            :,
            neutral_idx:neutral_idx + 1,
        ] = neutral_column

        return map_au2emo, map_emo2au

    def _gate(self, x):
        return torch.sigmoid(x)

    def forward(self, x, y):
        if x.ndim != 5 or y.ndim != 5:
            raise ValueError("SSM inputs must be five-dimensional clips.")
        if (
            x.shape[1] != 16
            or y.shape[1] != 16
            or x.shape[2] != 3
            or y.shape[2] != 3
            or x.shape[3:] != y.shape[3:]
        ):
            raise ValueError(
                "SSM requires matching 16-frame RGB clip shapes."
            )
        # Encode and normalize the learned emotion and AU prompts.
        prompts1 = self.prompt_learner1()
        tokenized_prompts1 = self.tokenized_prompts1
        text_features1 = self.text_encoder(
            prompts1,
            tokenized_prompts1,
        )
        text_features1 = (
            text_features1
            / text_features1.norm(
                dim=-1,
                keepdim=True,
            )
        )

        prompts2 = self.prompt_learner2()
        tokenized_prompts2 = self.tokenized_prompts2
        text_features2 = self.text_encoder(
            prompts2,
            tokenized_prompts2,
        )
        text_features2 = (
            text_features2
            / text_features2.norm(
                dim=-1,
                keepdim=True,
            )
        )

        # Exchange semantic evidence through the learned bidirectional maps.
        map_au2emo, map_emo2au = self._compute_maps()
        self.last_map_au2emo = map_au2emo.detach()
        self.last_map_emo2au = map_emo2au.detach()

        aug_text_features1 = (
            map_au2emo @ text_features2
        )
        a = self._gate(self.alpha)
        text_features1_z = (
            text_features1
            + a * aug_text_features1
        )
        text_features1_z = (
            text_features1_z
            / text_features1_z.norm(
                dim=-1,
                keepdim=True,
            )
        )

        aug_text_features2 = (
            map_emo2au @ text_features1
        )
        b = self._gate(self.beta)
        text_features2_z = (
            text_features2
            + b * aug_text_features2
        )
        text_features2_z = (
            text_features2_z
            / text_features2_z.norm(
                dim=-1,
                keepdim=True,
            )
        )

        B1, T, C, H, W = x.shape
        B2, _, _, _, _ = y.shape

        # Aggregate the expression clip into one temporally encoded feature.
        x_flat = x.view(-1, C, H, W)
        combined_fer = self.encoder(
            x_flat.to(self.dtype)
        )
        dfer_feat = combined_fer.view(
            B1,
            T,
            -1,
        )
        dfer_feat = self.temporal_net(dfer_feat)
        dfer_logits_1 = self.dfer_head(dfer_feat)
        dfer_feat = (
            dfer_feat
            / dfer_feat.norm(
                dim=-1,
                keepdim=True,
            )
        )

        dfer_logits_2 = (
            dfer_feat @ text_features1_z.t()
            / 0.01
        )
        dfer_logits = (
            dfer_logits_1
            + 0.1 * dfer_logits_2
        )

        # Retain one temporally contextualized prediction per AU frame.
        y_flat = y.view(-1, C, H, W)
        combined_au = self.encoder(
            y_flat.to(self.dtype)
        )
        combined_au = combined_au.reshape(
            B2,
            16,
            512,
        )
        combined_au = self.temporal_net_au(
            combined_au
        )
        combined_au = combined_au.reshape(-1, 512)
        au_logits_flat_1 = self.au_head(combined_au)
        combined_au = (
            combined_au
            / combined_au.norm(
                dim=-1,
                keepdim=True,
            )
        )
        au_logits_flat_2 = (
            combined_au @ text_features2_z.t()
            / 0.01
        )
        au_logits_flat = (
            au_logits_flat_1
            + 0.1 * au_logits_flat_2
        )

        # Apply the legacy smoother to consecutive two-frame AU groups.
        au_logits = au_logits_flat.view(
            B2 * 16 // 2,
            2,
            8,
        )
        smooth_in = au_logits.permute(0, 2, 1)
        smooth_out = self.au_smooth(smooth_in)
        au_logits_sm = smooth_out.permute(0, 2, 1)
        au_logits_sm = au_logits_sm.reshape(-1, 8)

        return (
            dfer_logits,
            0.1 * dfer_logits_2,
            au_logits_sm,
            0.1 * au_logits_flat_2,
            map_au2emo,
            map_emo2au,
        )
__all__ = [
    "DisfaSSM",
]
