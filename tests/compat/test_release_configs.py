# Validate all six release configs and their official fold combinations.
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import evaluate as evaluate_entry
import ssm.runner
import train as train_entry

from ssm.config import (
    configured_fold_pairs,
    load_config,
    validate_config,
    validate_fold_pair,
)
from train import all_fold_pairs


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CONFIGS = {
    "bp4d_dfew.json": ("DFEW", "BP4D", [(5, 1)]),
    "disfa_dfew.json": ("DFEW", "DISFA", [(5, 1)]),
    "bp4d_ferv39k.json": ("FERV39K", "BP4D", [(1, 1)]),
    "disfa_ferv39k.json": ("FERV39K", "DISFA", [(1, 1)]),
    "bp4d_mafw.json": ("MAFW", "BP4D", [(5, 1)]),
    "disfa_mafw.json": ("MAFW", "DISFA", [(5, 1)]),
}

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def test_release_configs_validate_and_select_the_expected_pair():
    for filename, expected in EXPECTED_CONFIGS.items():
        config = load_config(REPOSITORY_ROOT / "configs" / filename)
        actual = (
            config["data"]["emotion"]["name"],
            config["data"]["au"]["name"],
            configured_fold_pairs(config),
        )
        assert actual == expected
        assert config["runtime"]["gpu_ids"] == [0, 1, 2]
        expected_batch_size = 10 if expected[0] == "MAFW" else 12
        assert config["data"]["emotion"]["batch_size"] == expected_batch_size
        expected_classes = 11 if expected[0] == "MAFW" else 7
        assert config["model"]["emotion_classes"] == expected_classes
        emotion_config = config["data"]["emotion"]
        for key in (
            "normalization",
            "train_normalization",
            "eval_normalization",
        ):
            if key in emotion_config:
                assert emotion_config[key]["mean"] == CLIP_MEAN
                assert emotion_config[key]["std"] == CLIP_STD
        assert config["data"]["au"]["normalization"]["mean"] == CLIP_MEAN
        assert config["data"]["au"]["normalization"]["std"] == CLIP_STD
        au_paths = config["data"]["au"]
        if expected[1] == "BP4D":
            assert au_paths["train_list"].endswith("train_{au_fold}f_data.list")
            assert au_paths["test_list"].endswith("test_{au_fold}f_data.list")
        else:
            assert au_paths["train_list"].endswith("DISFA_train{au_fold}.json")
            assert au_paths["test_list"].endswith("DISFA_test{au_fold}.json")


def test_effective_legacy_aliases_cannot_bypass_reproduction_settings():
    source = load_config(REPOSITORY_ROOT / "configs" / "bp4d_dfew.json")
    mutations = [
        ("runtime.seed", ("runtime", "seed", 99)),
        ("runtime.workers", ("runtime", "workers", 0)),
        (
            "optimization.epochs",
            ("optimization", "epochs", 2),
        ),
        (
            "optimization.milestones",
            ("optimization", "milestones", [1]),
        ),
        (
            "model.clip_backbone",
            ("model", "clip_backbone", "RN50"),
        ),
        (
            "model.context_tokens",
            ("model", "context_tokens", 3),
        ),
        (
            "model.class_token_position",
            ("model", "class_token_position", "front"),
        ),
        (
            "evaluation.adaptive_vote.half_window_max",
            (
                "evaluation",
                "adaptive_vote",
                {
                    "enabled": True,
                    "half_window_min": 0,
                    "half_window_max": 9,
                    "search_all_vote_requirements": True,
                    "segment_safe": True,
                    "complete_frames": True,
                    "tie_reference": {
                        "half_window": 3,
                        "positive_ratio": 0.3,
                        "threshold": 0.5,
                    },
                },
            ),
        ),
        (
            "runtime.print_frequency",
            ("runtime", "print_frequency", 0),
        ),
    ]

    for setting, (section, key, value) in mutations:
        config = copy.deepcopy(source)
        config[section][key] = value
        with pytest.raises(ValueError, match=setting):
            validate_config(config)

    config = copy.deepcopy(source)
    config["runtime"]["gpu_ids"] = [0, 0, 1]
    with pytest.raises(ValueError, match="three unique"):
        validate_config(config)


def test_only_official_expression_and_au_fold_ids_are_accepted():
    ferv = load_config(REPOSITORY_ROOT / "configs" / "bp4d_ferv39k.json")
    with pytest.raises(ValueError, match="FERV39K fold"):
        validate_fold_pair(ferv, 2, 1)

    mafw = load_config(REPOSITORY_ROOT / "configs" / "disfa_mafw.json")
    validate_fold_pair(mafw, 1, 1)
    validate_fold_pair(mafw, 5, 3)
    with pytest.raises(ValueError, match="MAFW fold"):
        validate_fold_pair(mafw, 0, 1)
    with pytest.raises(ValueError, match="AU fold"):
        validate_fold_pair(mafw, 5, 4)


def test_all_fold_order_matches_the_original_nested_loops():
    assert all_fold_pairs("DFEW") == [
        {"emotion_fold": emotion_fold, "au_fold": au_fold}
        for emotion_fold in [5, 4, 3, 2, 1]
        for au_fold in [1, 2, 3]
    ]
    assert all_fold_pairs("MAFW") == all_fold_pairs("DFEW")
    assert all_fold_pairs("FERV39K") == [
        {"emotion_fold": 1, "au_fold": au_fold} for au_fold in [1, 2, 3]
    ]


def test_evaluation_launchers_forward_fold_overrides():
    scripts = REPOSITORY_ROOT / "scripts"
    for path in scripts.glob("evaluate_*.sh"):
        source = path.read_text(encoding="utf-8")
        assert "if [[ $# -lt 1 ]]" in source
        assert 'CHECKPOINT="$PWD/$CHECKPOINT"' in source
        assert "\nshift\n" in source
        assert '"$@"' in source


def test_training_entry_preserves_existing_cuda_visibility(
    monkeypatch,
    capsys,
):
    config = {"runtime": {"gpu_ids": [0, 1, 2]}}
    cli = SimpleNamespace(
        config="unused.json",
        emotion_root=None,
        au_root=None,
        output_dir=None,
        emotion_fold=None,
        au_fold=None,
        all_folds=False,
        device=None,
    )
    monkeypatch.setattr(train_entry, "parse_args", lambda: cli)
    monkeypatch.setattr(
        train_entry,
        "load_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        train_entry,
        "apply_overrides",
        lambda loaded, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        train_entry,
        "validate_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        ssm.runner,
        "train_all",
        lambda _config: [],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6,7")

    train_entry.main()

    assert train_entry.os.environ["CUDA_VISIBLE_DEVICES"] == "5,6,7"
    capsys.readouterr()


def test_evaluation_entry_preserves_existing_cuda_visibility(
    monkeypatch,
    capsys,
):
    config = {
        "runtime": {"gpu_ids": [0, 1, 2]},
        "experiment": {"emotion_fold": 5, "au_fold": 1},
    }
    cli = SimpleNamespace(
        config="unused.json",
        checkpoint="unused.pth",
        emotion_root=None,
        au_root=None,
        emotion_fold=None,
        au_fold=None,
        device=None,
    )
    monkeypatch.setattr(evaluate_entry, "parse_args", lambda: cli)
    monkeypatch.setattr(
        evaluate_entry,
        "load_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        evaluate_entry,
        "apply_overrides",
        lambda loaded, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        evaluate_entry,
        "validate_fold_pair",
        lambda _config, _emotion_fold, _au_fold: None,
    )
    monkeypatch.setattr(
        ssm.runner,
        "evaluate_checkpoint",
        lambda *_args: {},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6,7")

    evaluate_entry.main()

    assert evaluate_entry.os.environ["CUDA_VISIBLE_DEVICES"] == "5,6,7"
    capsys.readouterr()
