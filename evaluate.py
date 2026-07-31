import argparse
import json
import os

from ssm.config import apply_overrides, load_config, validate_fold_pair


# Evaluation uses the same config schema and fold identifiers as training.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate both expression and AU tasks from one joint "
            "SSM checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Joint model checkpoint. The primary best_emotion.pth "
            "checkpoint supports both tasks."
        ),
    )
    parser.add_argument("--emotion-fold", type=int)
    parser.add_argument("--au-fold", type=int)
    parser.add_argument("--emotion-root")
    parser.add_argument("--au-root")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    cli = parse_args()
    config = load_config(cli.config)
    # Override only machine-specific dataset locations at runtime.
    config = apply_overrides(
        config,
        emotion_root=cli.emotion_root,
        au_root=cli.au_root,
    )
    if cli.device is not None:
        config["runtime"]["device"] = cli.device
    emotion_fold = (
        cli.emotion_fold
        if cli.emotion_fold is not None
        else int(config["experiment"]["emotion_fold"])
    )
    au_fold = (
        cli.au_fold
        if cli.au_fold is not None
        else int(config["experiment"]["au_fold"])
    )
    validate_fold_pair(config, emotion_fold, au_fold)

    # Explicit config wins; otherwise preserve scheduler-provided visibility.
    visible_devices = config["runtime"].get("cuda_visible_devices")
    if visible_devices not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    elif (
        os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
        and config["runtime"].get("gpu_ids")
    ):
        visible_devices = ",".join(
            str(index) for index in config["runtime"]["gpu_ids"]
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)

    from ssm.runner import evaluate_checkpoint

    # A checkpoint is always evaluated on the explicitly resolved fold pair.
    result = evaluate_checkpoint(
        config,
        cli.checkpoint,
        emotion_fold,
        au_fold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
