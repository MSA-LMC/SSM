# Verify the enabled eleven-class MAFW semantics and model shapes.
from types import SimpleNamespace

import torch
from torch import nn

from ssm.models.bp4d import Bp4dSSM
from ssm.models.disfa import DisfaSSM
from ssm.semantics import (
    MAFW_EMOTION_LABELS,
    get_emotion_labels,
    get_task_descriptions,
)


BP4D_MAFW_PRIOR = torch.tensor(
    [
        [0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02],
        [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.02, 0.02, 1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02],
        [1.00, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02],
        [1.00, 1.00, 1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 1.00, 0.02, 0.02, 0.02, 0.02],
        [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02],
        [1.00, 0.02, 1.00, 0.02, 0.02, 0.02, 0.02, 0.02, 1.00, 0.02, 0.02, 0.02],
    ]
)

DISFA_MAFW_PRIOR = torch.tensor(
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
    ]
)


class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(2, 4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4, 2)


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeMlp()


class _FakeVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_FakeBlock() for _ in range(6)])


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _FakeVisionTransformer()

    def forward(self, images):
        values = images.mean(dim=(1, 2, 3)).unsqueeze(1)
        return values.repeat(1, 512)


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.token_embedding = nn.Embedding(49408, 2)
        self.transformer = nn.Identity()
        self.positional_embedding = nn.Parameter(torch.zeros(77, 2))
        self.ln_final = nn.LayerNorm(2)
        self.text_projection = nn.Parameter(torch.randn(2, 512))
        self.visual = _FakeVisual()


def _build_mafw_model(au_dataset):
    emotion_text, au_text = get_task_descriptions(
        "MAFW",
        au_dataset,
    )
    args = SimpleNamespace(
        contexts_number=8,
        class_specific_contexts="True",
        class_token_position="end",
        smooth_K=2,
        dataset="MAFW",
    )
    clip_model = _FakeClip()
    if au_dataset == "bp4d":
        return Bp4dSSM(
            emotion_text,
            au_text,
            clip_model,
            args,
        )
    return DisfaSSM(
        emotion_text,
        au_text,
        ["expression"] * 11,
        ["action unit"] * 8,
        clip_model,
        args,
    )


def test_mafw_semantics_match_the_eleven_class_order():
    assert get_emotion_labels("MAFW") == MAFW_EMOTION_LABELS
    assert MAFW_EMOTION_LABELS == [
        "happiness",
        "sadness",
        "neutral",
        "anger",
        "surprise",
        "disgust",
        "fear",
        "contempt",
        "anxiety",
        "helplessness",
        "disappointment",
    ]


def test_mafw_heads_priors_maps_and_forward_shapes():
    for au_dataset, expected_prior, au_count in (
        ("bp4d", BP4D_MAFW_PRIOR, 12),
        ("disfa", DISFA_MAFW_PRIOR, 8),
    ):
        torch.manual_seed(9)
        model = _build_mafw_model(au_dataset)
        model.eval()
        assert model.dfer_head.out_features == 11
        assert model.prompt_learner1.n_cls == 11
        assert torch.equal(model.init_map, expected_prior)
        assert model.delta_map1.shape == (11, au_count)
        assert model.delta_map2.shape == (au_count, 11)
        assert torch.count_nonzero(model.delta_map1) == 0
        assert torch.count_nonzero(model.delta_map2) == 0

        map_au2emo, map_emo2au = model._compute_maps()
        assert map_au2emo.shape == (11, au_count)
        assert map_emo2au.shape == (au_count, 11)
        assert torch.equal(map_au2emo[2], expected_prior[2])
        assert torch.equal(map_emo2au[:, 2], expected_prior.t()[:, 2])

        torch.manual_seed(10)
        emotion_frames = torch.randn(1, 16, 3, 2, 2)
        au_frames = torch.randn(1, 16, 3, 2, 2)
        with torch.no_grad():
            outputs = model(emotion_frames, au_frames)
        assert outputs[0].shape == (1, 11)
        assert outputs[1].shape == (1, 11)
        assert outputs[2].shape == (16, au_count)
        assert outputs[3].shape == (16, au_count)
        assert outputs[4].shape == (11, au_count)
        assert outputs[5].shape == (au_count, 11)
