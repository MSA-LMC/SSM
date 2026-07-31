# Freeze the seven-class model schema and deterministic forward contract.
import hashlib
from types import SimpleNamespace

import torch
from torch import nn

from ssm.models.bp4d import Bp4dSSM
from ssm.models.disfa import DisfaSSM
from ssm.semantics import (
    BP4D_AU_DESCRIPTIONS,
    BP4D_EMOTION_DESCRIPTIONS,
    DISFA_AU_DESCRIPTIONS,
    DISFA_EMOTION_DESCRIPTIONS,
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
        self.resblocks = nn.ModuleList(
            [_FakeBlock() for _ in range(6)]
        )


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


ARGS = SimpleNamespace(
    contexts_number=8,
    class_specific_contexts="True",
    class_token_position="end",
    temporal_layers=1,
    smooth_K=2,
)


def _schema_digest(model):
    payload = "".join(
        f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n"
        for name, tensor in model.state_dict().items()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parameter_order_digest(model):
    payload = "\n".join(
        name for name, _ in model.named_parameters()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_model(au_dataset):
    torch.manual_seed(12345)
    clip_model = _FakeClip()
    if au_dataset == "bp4d":
        return Bp4dSSM(
            BP4D_EMOTION_DESCRIPTIONS,
            BP4D_AU_DESCRIPTIONS,
            clip_model,
            ARGS,
        )
    return DisfaSSM(
        DISFA_EMOTION_DESCRIPTIONS,
        DISFA_AU_DESCRIPTIONS,
        ["expression"] * 7,
        ["action unit"] * 8,
        clip_model,
        ARGS,
    )


def test_seven_class_model_schema_parameter_order_and_outputs_are_frozen():
    cases = {
        "bp4d": {
            "schema": (
                "39542df4ca4d092362720468b8038da288db81183530c4e86d"
                "d64491e1078050"
            ),
            "parameters": (
                "8ed722ddeee43f2db029e3812843d6d7d6cd06c55eb0076b3"
                "b5d57ef4c41b9af"
            ),
            "shapes": (
                (1, 7),
                (1, 7),
                (16, 12),
                (16, 12),
                (7, 12),
                (12, 7),
            ),
        },
        "disfa": {
            "schema": (
                "1baf9de470355e4031bda5dcf36104d0a1be3224bd31030411"
                "cd8761ffc38e7a"
            ),
            "parameters": (
                "89767cddf95fd0d751eb5cc8413fb0f2784fbc59179d16907"
                "baf7e60626ff4e4"
            ),
            "shapes": (
                (1, 7),
                (1, 7),
                (16, 8),
                (16, 8),
                (7, 8),
                (8, 7),
            ),
        },
    }

    x = torch.linspace(-1, 1, 192).reshape(1, 16, 3, 2, 2)
    y = torch.linspace(1, -1, 192).reshape(1, 16, 3, 2, 2)

    for au_dataset, expected in cases.items():
        model = _build_model(au_dataset)
        assert _schema_digest(model) == expected["schema"]
        assert (
            _parameter_order_digest(model)
            == expected["parameters"]
        )
        assert model.dfer_head.out_features == 7

        model.eval()
        with torch.no_grad():
            outputs = model(x, y)
        assert tuple(tuple(output.shape) for output in outputs) == (
            expected["shapes"]
        )
        assert all(torch.isfinite(output).all() for output in outputs)
