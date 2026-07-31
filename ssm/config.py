import copy
import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


# Load, normalize, and validate one release configuration as a unit.
def load_config(path):
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["_config_path"] = str(config_path)
    normalize_config(config)
    validate_config(config)
    return config


def normalize_config(config):
    # Accept legacy aliases while exposing one canonical runtime schema.
    experiment = config["experiment"]
    emotion = config["data"]["emotion"]
    au = config["data"]["au"]
    model = config["model"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    paths = config["paths"]

    emotion.setdefault("name", experiment["emotion_dataset"])
    au.setdefault("name", experiment["au_dataset"])
    emotion.setdefault("root", paths.get("emotion_frame_root", ""))
    emotion.setdefault(
        "train_list",
        paths.get("emotion_train_manifest"),
    )
    emotion.setdefault(
        "test_list",
        paths.get("emotion_val_manifest"),
    )
    au.setdefault("root", paths.get("au_image_root", ""))
    au.setdefault("train_list", paths.get("au_train_manifest"))
    au.setdefault("test_list", paths.get("au_val_manifest"))

    prompt = model.get("prompt", {})
    temporal = model.get("temporal", {})
    model.setdefault("clip_backbone", model.get("clip_arch", "ViT-B/16"))
    model.setdefault(
        "context_tokens",
        prompt.get("contexts_number", 8),
    )
    model.setdefault(
        "class_specific_contexts",
        prompt.get("class_specific_contexts", "True"),
    )
    model.setdefault(
        "class_token_position",
        prompt.get("class_token_position", "end"),
    )
    model.setdefault("num_frames", temporal.get("num_frames", 16))
    model.setdefault("emotion_classes", emotion.get("num_classes", 7))
    model.setdefault(
        "temporal_layers",
        temporal.get("depth", 1),
    )
    model.setdefault(
        "disfa_smoothing_radius",
        temporal.get("au_depthwise_smoothing_half_window", 2),
    )
    model.setdefault("image_size", emotion.get("image_size", 224))

    optimization.setdefault("epochs", experiment.get("epochs", 30))
    optimization.setdefault(
        "milestones",
        optimization.get("scheduler", {}).get("milestones", []),
    )
    runtime.setdefault("seed", experiment.get("seed", 1))
    runtime.setdefault("workers", emotion.get("workers", 4))

    majority = evaluation.get("majority_vote", {})
    evaluation.setdefault(
        "voting_radius",
        majority.get("half_window", 3),
    )
    evaluation.setdefault(
        "voting_positive_ratio",
        majority.get("positive_ratio", 0.3),
    )
    paths.setdefault("clip_cache", paths.get("clip_cache_dir", ""))


def validate_config(config):
    # Reject unsupported dataset pairs and inconsistent class counts early.
    required = [
        "experiment",
        "data",
        "model",
        "optimization",
        "runtime",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(
            "Missing config sections: " + ", ".join(missing)
        )

    emotion = config["data"]["emotion"]
    au = config["data"]["au"]
    emotion_name = emotion["name"].upper()
    if emotion_name not in {"DFEW", "FERV39K", "MAFW"}:
        raise ValueError(
            f"Unsupported emotion dataset: {emotion['name']}"
        )
    if au["name"].lower() not in {"bp4d", "disfa"}:
        raise ValueError("AU dataset must be BP4D or DISFA.")
    if int(config["model"].get("num_frames", 16)) != 16:
        raise ValueError("Legacy-compatible SSM requires 16 frames.")
    expected_classes = 11 if emotion_name == "MAFW" else 7
    if int(emotion.get("num_classes", expected_classes)) != expected_classes:
        raise ValueError(
            f"{emotion_name} requires {expected_classes} classes."
        )
    if (
        int(
            config["model"].get(
                "emotion_classes",
                expected_classes,
            )
        )
        != expected_classes
    ):
        raise ValueError(
            f"{emotion_name} requires a {expected_classes}-class head."
        )
    if config.get("legacy_compatibility", False):
        validate_legacy_config(config)
    for emotion_fold, au_fold in configured_fold_pairs(config):
        validate_fold_pair(config, emotion_fold, au_fold)


def validate_legacy_config(config):
    # Freeze every setting that participates in numerical reproduction.
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    au_name = data["au"]["name"].lower()
    emotion_name = data["emotion"]["name"].upper()
    emotion_classes = 11 if emotion_name == "MAFW" else 7
    gpu_ids = runtime.get("gpu_ids", [])
    if (
        len(gpu_ids) != 3
        or any(
            not isinstance(index, int) or index < 0
            for index in gpu_ids
        )
        or len(set(gpu_ids)) != 3
    ):
        raise ValueError(
            "runtime.gpu_ids must contain three unique non-negative "
            "integers."
        )
    if int(runtime.get("print_frequency", 0)) <= 0:
        raise ValueError("runtime.print_frequency must be positive.")

    # These values mirror the successful paths in both original projects.
    expected = {
        "experiment.emotion_dataset": (
            str(experiment.get("emotion_dataset", "")).upper(),
            emotion_name,
        ),
        "experiment.au_dataset": (
            str(experiment.get("au_dataset", "")).lower(),
            au_name,
        ),
        "experiment.seed": (experiment.get("seed"), 1),
        "experiment.epochs": (experiment.get("epochs"), 30),
        "data.emotion.batch_size": (
            data["emotion"].get("batch_size"),
            12,
        ),
        "data.emotion.num_segments": (
            data["emotion"].get("num_segments"),
            16,
        ),
        "data.emotion.duration": (
            data["emotion"].get("duration"),
            1,
        ),
        "data.emotion.num_classes": (
            data["emotion"].get("num_classes"),
            emotion_classes,
        ),
        "data.emotion.shuffle_train": (
            data["emotion"].get("shuffle_train"),
            True,
        ),
        "data.emotion.drop_last_train": (
            data["emotion"].get("drop_last_train"),
            True,
        ),
        "data.emotion.drop_last_val": (
            data["emotion"].get("drop_last_val"),
            True,
        ),
        "data.au.batch_size": (
            data["au"].get("batch_size"),
            9 if au_name == "bp4d" else 8,
        ),
        "data.au.clip_len": (data["au"].get("clip_len"), 16),
        "data.au.stride": (data["au"].get("stride"), 16),
        "data.au.shuffle_train": (
            data["au"].get("shuffle_train"),
            True,
        ),
        "data.au.drop_last_train": (
            data["au"].get("drop_last_train"),
            True,
        ),
        "data.au.drop_last_val": (
            data["au"].get("drop_last_val"),
            True,
        ),
        "model.clip_arch": (model.get("clip_arch"), "ViT-B/16"),
        "model.clip_backbone": (
            model.get("clip_backbone"),
            "ViT-B/16",
        ),
        "model.clip_checkpoint": (
            model.get("clip_checkpoint") or None,
            None,
        ),
        "model.emotion_classes": (
            model.get("emotion_classes"),
            emotion_classes,
        ),
        "model.context_tokens": (
            model.get("context_tokens"),
            8,
        ),
        "model.class_specific_contexts": (
            model.get("class_specific_contexts"),
            "True",
        ),
        "model.class_token_position": (
            model.get("class_token_position"),
            "end",
        ),
        "model.num_frames": (model.get("num_frames"), 16),
        "model.temporal_layers": (
            model.get("temporal_layers"),
            1,
        ),
        "model.disfa_smoothing_radius": (
            model.get("disfa_smoothing_radius"),
            2,
        ),
        "model.image_size": (model.get("image_size"), 224),
        "model.prompt.contexts_number": (
            model["prompt"].get("contexts_number"),
            8,
        ),
        "model.prompt.class_specific_contexts": (
            model["prompt"].get("class_specific_contexts"),
            "True",
        ),
        "model.prompt.class_token_position": (
            model["prompt"].get("class_token_position"),
            "end",
        ),
        "model.temporal.num_frames": (
            model["temporal"].get("num_frames"),
            16,
        ),
        "model.temporal.depth": (
            model["temporal"].get("depth"),
            1,
        ),
        "model.temporal.heads": (
            model["temporal"].get("heads"),
            8,
        ),
        "model.temporal.mlp_dim": (
            model["temporal"].get("mlp_dim"),
            1024,
        ),
        "model.temporal.dim_head": (
            model["temporal"].get("dim_head"),
            64,
        ),
        "model.moe.replace_last_n_ffn": (
            model["moe"].get("replace_last_n_ffn"),
            6,
        ),
        "model.moe.private_experts": (
            model["moe"].get("private_experts"),
            4,
        ),
        "model.moe.expert_hidden_dim": (
            model["moe"].get("expert_hidden_dim"),
            512,
        ),
        "model.moe.top_k": (model["moe"].get("top_k"), 2),
        "model.moe.noise_std": (
            model["moe"].get("noise_std"),
            0.01,
        ),
        "model.semantic_mapping.alpha_init": (
            model["semantic_mapping"].get("alpha_init"),
            0.1,
        ),
        "model.semantic_mapping.beta_init": (
            model["semantic_mapping"].get("beta_init"),
            0.1,
        ),
        "model.semantic_mapping.epsilon": (
            model["semantic_mapping"].get("epsilon"),
            1e-8,
        ),
        "model.semantic_mapping.similarity_temperature": (
            model["semantic_mapping"].get(
                "similarity_temperature"
            ),
            0.01,
        ),
        "model.semantic_mapping.semantic_logit_weight": (
            model["semantic_mapping"].get("semantic_logit_weight"),
            0.1,
        ),
        "optimization.optimizer.weight_decay": (
            optimization["optimizer"].get("weight_decay"),
            0.0001,
        ),
        "optimization.mixup.alpha": (
            optimization["mixup"].get("alpha"),
            0.4,
        ),
        "optimization.mixup.enabled": (
            optimization["mixup"].get("enabled"),
            True,
        ),
        "optimization.mixup.scope": (
            optimization["mixup"].get("scope"),
            "flattened_au_frames",
        ),
        "optimization.ema.decay": (
            optimization["ema"].get("decay"),
            0.99,
        ),
        "optimization.ema.enabled": (
            optimization["ema"].get("enabled"),
            True,
        ),
        "optimization.loss_balance.moving_mean_momentum": (
            optimization["loss_balance"].get(
                "moving_mean_momentum"
            ),
            0.95,
        ),
        "optimization.loss_balance.au_weight": (
            optimization["loss_balance"].get("au_weight"),
            2.0,
        ),
        "optimization.loss_balance.epsilon": (
            optimization["loss_balance"].get("epsilon"),
            1e-8,
        ),
        "optimization.loss_balance.denominator": (
            optimization["loss_balance"].get("denominator"),
            3.0,
        ),
        "optimization.gradient_clip_norm": (
            optimization.get("gradient_clip_norm"),
            1.0,
        ),
        "optimization.epochs": (
            optimization.get("epochs"),
            30,
        ),
        "optimization.milestones": (
            optimization.get("milestones"),
            [10, 25] if au_name == "bp4d" else [15, 25],
        ),
        "runtime.seed": (runtime.get("seed"), 1),
        "runtime.workers": (runtime.get("workers"), 4),
        "runtime.cudnn_benchmark": (
            runtime.get("cudnn_benchmark"),
            False,
        ),
        "runtime.cudnn_deterministic": (
            runtime.get("cudnn_deterministic"),
            True,
        ),
        "runtime.data_parallel": (
            runtime.get("data_parallel"),
            True,
        ),
        "runtime.pin_memory": (
            runtime.get("pin_memory"),
            True,
        ),
        "runtime.gpu_count": (
            len(gpu_ids),
            3,
        ),
        "evaluation.legacy_validation_pairing": (
            evaluation.get("legacy_validation_pairing"),
            "zip_shortest",
        ),
        "evaluation.majority_vote.half_window": (
            evaluation["majority_vote"].get("half_window"),
            3,
        ),
        "evaluation.majority_vote.positive_ratio": (
            evaluation["majority_vote"].get("positive_ratio"),
            0.3,
        ),
        "evaluation.majority_vote.enabled": (
            evaluation["majority_vote"].get("enabled"),
            True,
        ),
        "evaluation.voting_radius": (
            evaluation.get("voting_radius"),
            3,
        ),
        "evaluation.voting_positive_ratio": (
            evaluation.get("voting_positive_ratio"),
            0.3,
        ),
        "evaluation.au_threshold.default": (
            evaluation["au_threshold"].get("default"),
            0.5,
        ),
        "evaluation.au_threshold.search_start": (
            evaluation["au_threshold"].get("search_start"),
            0.01,
        ),
        "evaluation.au_threshold.search_stop": (
            evaluation["au_threshold"].get("search_stop"),
            0.99,
        ),
        "evaluation.au_threshold.search_step": (
            evaluation["au_threshold"].get("search_step"),
            0.01,
        ),
    }

    learning_rates = optimization["optimizer"]["learning_rates"]
    expected.update(
        {
            "optimization.lr.encoder": (
                learning_rates.get("encoder"),
                0.000001,
            ),
            "optimization.lr.delta_map1": (
                learning_rates.get("delta_map1"),
                0.01,
            ),
            "optimization.lr.delta_map2": (
                learning_rates.get("delta_map2"),
                0.01,
            ),
            "optimization.lr.other": (
                learning_rates.get("other"),
                0.0001,
            ),
        }
    )

    expected_au_ids = (
        [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        if au_name == "bp4d"
        else [1, 2, 4, 6, 9, 12, 25, 26]
    )
    expected["data.au.au_ids"] = (
        data["au"].get("au_ids"),
        expected_au_ids,
    )

    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "Legacy-compatible settings cannot be changed:\n"
            + "\n".join(mismatches)
        )


def apply_overrides(
    config,
    emotion_root=None,
    au_root=None,
    output_dir=None,
):
    # Runtime path overrides must not mutate the loaded configuration object.
    result = copy.deepcopy(config)
    if emotion_root is not None:
        result["data"]["emotion"]["root"] = emotion_root
    if au_root is not None:
        result["data"]["au"]["root"] = au_root
    if output_dir is not None:
        result["paths"]["output_dir"] = output_dir
    return result


def format_template(value, emotion_fold, au_fold):
    # Support both unified and legacy placeholder names.
    if value is None:
        return None
    return str(value).format(
        emotion_fold=emotion_fold,
        au_fold=au_fold,
        dfer_fold=emotion_fold,
        fold=au_fold,
    )


def validate_fold_pair(config, emotion_fold, au_fold):
    # FERV39K has one expression split; DFEW and MAFW each have five.
    emotion_name = config["data"]["emotion"]["name"].upper()
    emotion_fold = int(emotion_fold)
    au_fold = int(au_fold)
    if emotion_name == "FERV39K":
        valid_emotion_folds = {1}
    else:
        valid_emotion_folds = {1, 2, 3, 4, 5}
    if emotion_fold not in valid_emotion_folds:
        raise ValueError(
            f"Invalid {emotion_name} fold: {emotion_fold}."
        )
    if au_fold not in {1, 2, 3}:
        raise ValueError(f"Invalid AU fold: {au_fold}.")


def resolve_path(value, emotion_fold=None, au_fold=None):
    # Relative release paths are always anchored at the repository root.
    if value in (None, ""):
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if emotion_fold is not None and au_fold is not None:
        expanded = format_template(
            expanded,
            emotion_fold,
            au_fold,
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def configured_fold_pairs(config):
    # Explicit pairs take precedence over Cartesian fold declarations.
    experiment = config["experiment"]
    explicit_pairs = experiment.get("fold_pairs")
    if explicit_pairs:
        return [
            (
                int(item["emotion_fold"]),
                int(item["au_fold"]),
            )
            for item in explicit_pairs
        ]

    if (
        "emotion_fold" in experiment
        and "au_fold" in experiment
        and "emotion_folds" not in experiment
        and "au_folds" not in experiment
    ):
        return [
            (
                int(experiment["emotion_fold"]),
                int(experiment["au_fold"]),
            )
        ]

    emotion_folds = experiment.get(
        "emotion_folds",
        config["data"]["emotion"].get("folds", [1]),
    )
    au_folds = experiment.get(
        "au_folds",
        config["data"]["au"].get("folds", [1]),
    )
    return [
        (int(emotion_fold), int(au_fold))
        for emotion_fold in emotion_folds
        for au_fold in au_folds
    ]
