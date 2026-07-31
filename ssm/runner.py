import datetime
import json
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn

from ssm.config import configured_fold_pairs, resolve_path
from ssm.data.bp4d import Bp4dSequenceDataset
from ssm.data.disfa import build_disfa_dataset
from ssm.data.emotion import (
    build_emotion_evaluation_dataset,
    build_emotion_training_dataset,
)
from ssm.evaluation.metrics import (
    evaluate_action_units,
    evaluate_emotions,
)
from ssm.models import Bp4dSSM, DisfaSSM
from ssm.semantics import (
    get_emotion_labels,
    get_task_descriptions,
)
from ssm.third_party import openai_clip
from ssm.training.legacy import (
    LegacyCurveRecorder,
    ModelEMA,
    build_legacy_optimizer,
    set_reproducible_seed,
    train_joint_epoch,
    validate_joint,
)


def _config_value(section, *names, default=None):
    for name in names:
        if name in section:
            return section[name]
    return default


def build_runtime_args(config):
    # Bridge the unified JSON schema to legacy model and loader arguments.
    emotion = config["data"]["emotion"]
    model = config["model"]
    runtime = config["runtime"]
    optimization = config["optimization"]

    class_specific = model.get("class_specific_contexts", True)
    if isinstance(class_specific, bool):
        class_specific = "True" if class_specific else "False"

    return SimpleNamespace(
        dataset=emotion["name"].upper(),
        workers=int(runtime.get("workers", 4)),
        epochs=int(optimization.get("epochs", 30)),
        batch_size=int(emotion.get("batch_size", 12)),
        print_freq=int(runtime.get("print_frequency", 100)),
        milestones=[
            int(value)
            for value in optimization.get("milestones", [])
        ],
        contexts_number=int(model.get("context_tokens", 8)),
        class_token_position=model.get(
            "class_token_position",
            "end",
        ),
        class_specific_contexts=class_specific,
        seed=int(runtime.get("seed", 1)),
        temporal_layers=int(model.get("temporal_layers", 1)),
        smooth_K=int(model.get("disfa_smoothing_radius", 2)),
        input_size=int(model.get("image_size", 224)),
        root_path="",
    )


def _resolve_clip_source(config):
    model_config = config["model"]
    checkpoint = model_config.get("clip_checkpoint")
    if checkpoint:
        return str(resolve_path(checkpoint))
    return model_config.get("clip_backbone", "ViT-B/16")


def _validate_runtime_device(config, device):
    if device.type != "cuda":
        return
    runtime = config["runtime"]
    if (
        config.get("legacy_compatibility", False)
        and device.index is not None
    ):
        raise ValueError(
            "Legacy-compatible CUDA runs require the unindexed 'cuda' "
            "device."
        )
    if runtime.get("data_parallel", False):
        expected = len(runtime.get("gpu_ids", []))
        actual = torch.cuda.device_count()
        if actual != expected:
            raise RuntimeError(
                "The configured DataParallel topology requires "
                f"{expected} visible GPUs, but found {actual}."
            )


def build_joint_model(config, args, device):
    au_name = config["data"]["au"]["name"].lower()
    emotion_name = config["data"]["emotion"]["name"].upper()
    # Prompt order must stay aligned with class and AU output indices.
    emotion_text, au_text = get_task_descriptions(
        emotion_name,
        au_name,
    )
    clip_source = _resolve_clip_source(config)
    cache = config["paths"].get("clip_cache")

    if cache:
        clip_model, _ = openai_clip.load(
            clip_source,
            device="cpu",
            download_root=str(resolve_path(cache)),
        )
    else:
        clip_model, _ = openai_clip.load(
            clip_source,
            device="cpu",
        )

    if au_name == "bp4d":
        model = Bp4dSSM(
            input_text1=emotion_text,
            input_text2=au_text,
            clip_model=clip_model,
            args=args,
        )
    elif au_name == "disfa":
        model = DisfaSSM(
            input_text1=emotion_text,
            input_text2=au_text,
            input_text3=["expression"] * len(emotion_text),
            input_text4=["action unit"] * len(au_text),
            clip_model=clip_model,
            args=args,
        )
    else:
        raise ValueError(f"Unsupported AU dataset: {au_name}")

    # Match the original policy: train all parameters except text encoders.
    for name, param in model.named_parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad = False

    # An unindexed CUDA device selects the legacy three-GPU DataParallel path.
    if device.type == "cuda":
        if device.index is None:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    return model


def _template_path(section, key, emotion_fold, au_fold):
    value = _config_value(
        section,
        key,
        key.replace("_list", "_annotation"),
    )
    if value is None:
        raise ValueError(f"Missing data path setting: {key}")
    return resolve_path(value, emotion_fold, au_fold)


def build_data_loaders(
    config,
    args,
    emotion_fold,
    au_fold,
):
    # Construct expression loaders first to preserve RNG initialization order.
    emotion_config = config["data"]["emotion"]
    au_config = config["data"]["au"]
    emotion_root = resolve_path(emotion_config.get("root"))

    emotion_train_list = _template_path(
        emotion_config,
        "train_list",
        emotion_fold,
        au_fold,
    )
    emotion_test_list = _template_path(
        emotion_config,
        "test_list",
        emotion_fold,
        au_fold,
    )
    if not emotion_train_list.is_file():
        raise FileNotFoundError(
            f"Emotion training split not found: {emotion_train_list}"
        )
    if not emotion_test_list.is_file():
        raise FileNotFoundError(
            f"Emotion test split not found: {emotion_test_list}"
        )

    train_data1 = build_emotion_training_dataset(
        list_file=str(emotion_train_list),
        num_segments=16,
        duration=1,
        image_size=224,
        args=args,
        root=str(emotion_root) if emotion_root else None,
    )
    test_data1 = build_emotion_evaluation_dataset(
        list_file=str(emotion_test_list),
        num_segments=16,
        duration=1,
        image_size=224,
        root=str(emotion_root) if emotion_root else None,
    )

    train_loader1 = torch.utils.data.DataLoader(
        train_data1,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader1 = torch.utils.data.DataLoader(
        test_data1,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    au_name = au_config["name"].lower()
    au_train_list = _template_path(
        au_config,
        "train_list",
        emotion_fold,
        au_fold,
    )
    au_test_list = _template_path(
        au_config,
        "test_list",
        emotion_fold,
        au_fold,
    )
    if not au_train_list.is_file():
        raise FileNotFoundError(
            f"AU training split not found: {au_train_list}"
        )
    if not au_test_list.is_file():
        raise FileNotFoundError(
            f"AU test split not found: {au_test_list}"
        )

    au_root = resolve_path(au_config.get("root"))
    if au_root is None:
        raise ValueError("The AU image root must be set in the config.")

    # BP4D and DISFA retain their distinct clip and transform implementations.
    if au_name == "bp4d":
        train_data2 = Bp4dSequenceDataset(
            label_file=str(au_train_list),
            root=str(au_root),
            is_train=True,
            skip_invalid=True,
        )
        test_data2 = Bp4dSequenceDataset(
            label_file=str(au_test_list),
            root=str(au_root),
            is_train=False,
            skip_invalid=True,
        )
    else:
        args.root_path = str(au_root)
        train_data2 = build_disfa_dataset(
            str(au_train_list),
            is_train=True,
            args=args,
        )
        test_data2 = build_disfa_dataset(
            str(au_test_list),
            is_train=False,
            args=args,
        )

    au_batch_size = int(
        au_config.get(
            "batch_size",
            9 if au_name == "bp4d" else 8,
        )
    )
    train_loader2 = torch.utils.data.DataLoader(
        train_data2,
        batch_size=au_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader2 = torch.utils.data.DataLoader(
        test_data2,
        batch_size=au_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    loaders = {
        "emotion training": train_loader1,
        "emotion validation": val_loader1,
        "AU training": train_loader2,
        "AU validation": val_loader2,
    }
    empty = [
        name for name, loader in loaders.items() if len(loader) == 0
    ]
    if empty:
        raise RuntimeError(
            "No complete batch is available for: "
            + ", ".join(empty)
            + ". Check the split size and drop_last batch size."
        )
    return train_loader1, val_loader1, train_loader2, val_loader2


def check_data_paths(config, emotion_fold, au_fold):
    # Fail before model construction when a selected fold cannot be resolved.
    emotion = config["data"]["emotion"]
    au = config["data"]["au"]
    checks = [
        (
            "emotion frame root",
            resolve_path(emotion.get("root")),
            "directory",
        ),
        (
            "emotion training split",
            _template_path(
                emotion,
                "train_list",
                emotion_fold,
                au_fold,
            ),
            "file",
        ),
        (
            "emotion test split",
            _template_path(
                emotion,
                "test_list",
                emotion_fold,
                au_fold,
            ),
            "file",
        ),
        (
            "AU image root",
            resolve_path(au.get("root")),
            "directory",
        ),
        (
            "AU training split",
            _template_path(
                au,
                "train_list",
                emotion_fold,
                au_fold,
            ),
            "file",
        ),
        (
            "AU test split",
            _template_path(
                au,
                "test_list",
                emotion_fold,
                au_fold,
            ),
            "file",
        ),
    ]
    invalid = []
    for label, path, expected_type in checks:
        valid = (
            path is not None
            and (
                path.is_dir()
                if expected_type == "directory"
                else path.is_file()
            )
        )
        if not valid:
            invalid.append(
                f"{label} ({expected_type}): {path}"
            )
    if invalid:
        raise FileNotFoundError(
            "Required data paths are missing or have the wrong type:\n"
            + "\n".join(invalid)
        )


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def _output_paths(config, emotion_fold, au_fold, timestamp):
    output_root = resolve_path(
        config["paths"].get("output_dir", "outputs"),
        emotion_fold,
        au_fold,
    )
    emotion_name = config["data"]["emotion"]["name"].lower()
    au_name = config["data"]["au"]["name"].lower()
    # Include both fold IDs so multi-fold runs never share checkpoints.
    run_name = (
        f"{emotion_name}_{au_name}_"
        f"emotion{emotion_fold}_au{au_fold}_{timestamp}"
    )
    run_dir = output_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log": run_dir / "train.log",
        "curve": run_dir / "curve.png",
        "confusion": run_dir / "confusion_matrix.png",
        "checkpoint": checkpoint_dir / "checkpoint.pth",
        "best_emotion": checkpoint_dir / "best_emotion.pth",
        "best_au": checkpoint_dir / "best_au.pth",
        "summary": run_dir / "summary.json",
    }


def save_checkpoint(
    state,
    is_best,
    checkpoint_path,
    best_checkpoint_path,
):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def _load_checkpoint_file(checkpoint_path, device):
    # Temporarily expose legacy RecorderMeter symbols for old pickle payloads.
    restored_symbols = []
    for module_name in ("__main__", "main", "main1"):
        module = sys.modules.get(module_name)
        created = module is None
        if created:
            module = ModuleType(module_name)
            sys.modules[module_name] = module
        previous = getattr(module, "RecorderMeter", None)
        existed = hasattr(module, "RecorderMeter")
        module.RecorderMeter = LegacyCurveRecorder
        restored_symbols.append(
            (module_name, module, created, existed, previous)
        )

    try:
        if device.type == "cuda":
            if device.index is None:
                return torch.load(
                    checkpoint_path,
                    weights_only=False,
                )
            return torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    finally:
        for (
            module_name,
            module,
            created,
            existed,
            previous,
        ) in reversed(restored_symbols):
            if created:
                sys.modules.pop(module_name, None)
            elif existed:
                module.RecorderMeter = previous
            else:
                delattr(module, "RecorderMeter")


def load_checkpoint_state(model, checkpoint_path, device):
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )

    # Retry only to reconcile DataParallel's optional "module." prefix.
    try:
        model.load_state_dict(state, strict=True)
        return checkpoint
    except RuntimeError:
        model_has_module = all(
            key.startswith("module.")
            for key in model.state_dict()
        )
        state_has_module = all(
            key.startswith("module.")
            for key in state
        )
        if state_has_module and not model_has_module:
            state = {
                key[len("module.") :]: value
                for key, value in state.items()
            }
        elif model_has_module and not state_has_module:
            state = {
                "module." + key: value
                for key, value in state.items()
            }
        model.load_state_dict(state, strict=True)
        return checkpoint


def _write_config_snapshot(config, path):
    clean = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(clean, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _append_best_metrics(
    log_path,
    max_f1,
    best_f1_epoch,
    max_auc,
    best_auc_epoch,
):
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(
            f"best F1: {max_f1:.4f} at epoch {best_f1_epoch}\n"
        )
        file.write(
            f"best AUC: {max_auc:.4f} at epoch {best_auc_epoch}\n"
        )


def train_fold_pair(
    config,
    emotion_fold,
    au_fold,
    timestamp,
):
    args = build_runtime_args(config)
    # Reset all RNGs independently for every requested fold pair.
    set_reproducible_seed(args.seed)

    runtime = config["runtime"]
    device = torch.device(runtime.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the selected config.")
    _validate_runtime_device(config, device)

    check_data_paths(config, emotion_fold, au_fold)
    paths = _output_paths(
        config,
        emotion_fold,
        au_fold,
        timestamp,
    )
    _write_config_snapshot(config, paths["run_dir"] / "config.json")
    with paths["log"].open("a", encoding="utf-8") as file:
        file.write(
            f"emotion_fold={emotion_fold}\n"
            f"au_fold={au_fold}\n"
        )

    best_acc = 0
    recorder = LegacyCurveRecorder(args.epochs)
    model = build_joint_model(config, args, device)
    core_model = _model_core(model)
    ema = ModelEMA(core_model, decay=0.99)

    if runtime.get("print_parameters", False):
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    if device.type == "cuda":
        if device.index is None:
            criterion1 = nn.CrossEntropyLoss().cuda()
            criterion2 = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion1 = nn.CrossEntropyLoss().to(device)
            criterion2 = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion1 = nn.CrossEntropyLoss().to(device)
        criterion2 = nn.BCEWithLogitsLoss().to(device)

    optimizer = build_legacy_optimizer(core_model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=0.1,
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    (
        train_loader1,
        val_loader1,
        train_loader2,
        val_loader2,
    ) = build_data_loaders(
        config,
        args,
        emotion_fold,
        au_fold,
    )
    print(
        "loader batches:",
        len(train_loader1),
        len(train_loader2),
        len(val_loader1),
        len(val_loader2),
    )

    au_name = config["data"]["au"]["name"].lower()
    au_count = 12 if au_name == "bp4d" else 8
    emotion_count = int(
        config["data"]["emotion"]["num_classes"]
    )
    max_f1 = 0.0
    best_f1_epoch = -1
    max_auc = 0.0
    best_auc_epoch = -1

    for epoch in range(0, args.epochs):
        start_time = time.time()
        (
            train_acc,
            train_los,
            loss1,
            map_au2emo,
            map_emo2au,
        ) = train_joint_epoch(
            train_loader1,
            train_loader2,
            model,
            criterion1,
            criterion2,
            optimizer,
            epoch,
            args.print_freq,
            str(paths["log"]),
            au_name,
            au_count,
            device,
            ema,
        )
        print(
            "map_au2emo",
            torch.round(
                map_au2emo[:emotion_count, :au_count] * 10000
            )
            / 10000,
        )
        print(
            "map_emo2au",
            torch.round(
                map_emo2au[:au_count, :emotion_count] * 10000
            )
            / 10000,
        )

        # Preserve scheduler, EMA validation, checkpoint, and restore ordering.
        scheduler.step()
        ema.apply_shadow()

        val_acc, val_los = validate_joint(
            val_loader1,
            val_loader2,
            model,
            criterion1,
            args.print_freq,
            str(paths["log"]),
            device,
        )
        # Expression and AU checkpoints use their original task-specific rules.
        is_best = epoch == 0 or val_acc > best_acc
        if is_best:
            best_acc = val_acc
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "recorder": recorder,
            },
            is_best,
            paths["checkpoint"],
            paths["best_emotion"],
        )

        recorder.update(
            epoch,
            train_los,
            train_acc,
            val_los,
            val_acc,
        )
        recorder.plot_curve(paths["curve"])

        _, f1_mean, auc_mean = evaluate_action_units(
            val_loader1,
            val_loader2,
            model,
            device,
            au_name,
            au_count,
            K=int(config["evaluation"].get("voting_radius", 3)),
            p=float(
                config["evaluation"].get(
                    "voting_positive_ratio",
                    0.3,
                )
            ),
        )
        if epoch == 0 or f1_mean > max_f1:
            max_f1 = f1_mean
            best_f1_epoch = epoch
            torch.save(model.state_dict(), paths["best_au"])
        if epoch == 0 or auc_mean > max_auc:
            max_auc = auc_mean
            best_auc_epoch = epoch

        epoch_time = time.time() - start_time
        print("The best accuracy: {:.3f}".format(best_acc.item()))
        print("An epoch time: {:.2f}s".format(epoch_time))
        print(
            f"Current epoch {epoch} - "
            f"F1: {f1_mean:.4f}, AUC: {auc_mean:.4f}"
        )
        _append_best_metrics(
            paths["log"],
            max_f1,
            best_f1_epoch,
            max_auc,
            best_auc_epoch,
        )
        ema.restore()

    # Final UAR/WAR are reported from the best expression checkpoint.
    load_checkpoint_state(model, paths["best_emotion"], device)
    uar, war = evaluate_emotions(
        val_loader1,
        val_loader2,
        model,
        device,
        get_emotion_labels(
            config["data"]["emotion"]["name"]
        ),
        config["data"]["emotion"]["name"].upper(),
        emotion_fold,
        paths["confusion"],
        paths["log"],
    )
    summary = {
        "emotion_fold": emotion_fold,
        "au_fold": au_fold,
        "best_emotion_accuracy": float(best_acc),
        "best_au_f1": float(max_f1),
        "best_au_f1_epoch": best_f1_epoch,
        "best_au_auc": float(max_auc),
        "best_au_auc_epoch": best_auc_epoch,
        "uar": float(uar),
        "war": float(war),
        "best_emotion_checkpoint": str(paths["best_emotion"]),
        "best_au_checkpoint": str(paths["best_au"]),
    }
    with paths["summary"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def train_all(config):
    args = build_runtime_args(config)
    set_reproducible_seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
    results = []
    # Execute fold pairs sequentially with one shared run timestamp.
    for emotion_fold, au_fold in configured_fold_pairs(config):
        results.append(
            train_fold_pair(
                config,
                emotion_fold,
                au_fold,
                timestamp,
            )
        )
    return results


def evaluate_checkpoint(
    config,
    checkpoint_path,
    emotion_fold,
    au_fold,
):
    # Standalone evaluation reproduces both task metrics on one fold pair.
    args = build_runtime_args(config)
    set_reproducible_seed(args.seed)
    device = torch.device(config["runtime"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the selected config.")
    _validate_runtime_device(config, device)

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
    check_data_paths(config, emotion_fold, au_fold)
    model = build_joint_model(config, args, device)
    load_checkpoint_state(model, checkpoint_path, device)
    (
        train_loader1,
        val_loader1,
        train_loader2,
        val_loader2,
    ) = build_data_loaders(
        config,
        args,
        emotion_fold,
        au_fold,
    )
    au_name = config["data"]["au"]["name"].lower()
    au_count = 12 if au_name == "bp4d" else 8
    _, f1_mean, auc_mean = evaluate_action_units(
        val_loader1,
        val_loader2,
        model,
        device,
        au_name,
        au_count,
        K=int(config["evaluation"].get("voting_radius", 3)),
        p=float(
            config["evaluation"].get(
                "voting_positive_ratio",
                0.3,
            )
        ),
    )
    uar, war = evaluate_emotions(
        val_loader1,
        val_loader2,
        model,
        device,
        get_emotion_labels(
            config["data"]["emotion"]["name"]
        ),
        config["data"]["emotion"]["name"].upper(),
        emotion_fold,
    )
    return {
        "emotion_fold": emotion_fold,
        "au_fold": au_fold,
        "f1": float(f1_mean),
        "auc": float(auc_mean),
        "uar": float(uar),
        "war": float(war),
    }
