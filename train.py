import argparse
import json
import os

from ssm.config import apply_overrides, load_config, validate_config


# Keep the command-line surface shared by all six dataset combinations.
def parse_args():
    parser = argparse.ArgumentParser(
        description=("Jointly train SSM on an expression/AU dataset pair.")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--emotion-root")
    parser.add_argument("--au-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--emotion-fold", type=int)
    parser.add_argument("--au-fold", type=int)
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run every expression/AU fold pair as an independent joint run.",
    )
    parser.add_argument("--device")
    return parser.parse_args()


def all_fold_pairs(emotion_dataset):
    # Preserve the fold order used by the original BP4D and DISFA runs.
    name = emotion_dataset.upper()
    if name == "FERV39K":
        return [{"emotion_fold": 1, "au_fold": au_fold} for au_fold in [1, 2, 3]]
    if name in {"DFEW", "MAFW"}:
        return [
            {
                "emotion_fold": emotion_fold,
                "au_fold": au_fold,
            }
            for emotion_fold in [1, 2, 3, 4, 5]
            for au_fold in [1, 2, 3]
        ]
    raise ValueError(f"Unsupported expression dataset: {emotion_dataset}")


def main():
    cli = parse_args()
    config = load_config(cli.config)
    # Local dataset roots can be supplied without editing release configs.
    config = apply_overrides(
        config,
        emotion_root=cli.emotion_root,
        au_root=cli.au_root,
        output_dir=cli.output_dir,
    )
    if cli.all_folds and (cli.emotion_fold is not None or cli.au_fold is not None):
        raise ValueError("--all-folds cannot be combined with fold overrides.")
    if cli.all_folds:
        config["experiment"]["fold_pairs"] = all_fold_pairs(
            config["experiment"]["emotion_dataset"]
        )
    elif cli.emotion_fold is not None or cli.au_fold is not None:
        if cli.emotion_fold is None or cli.au_fold is None:
            raise ValueError("--emotion-fold and --au-fold must be provided together.")
        config["experiment"]["fold_pairs"] = [
            {
                "emotion_fold": cli.emotion_fold,
                "au_fold": cli.au_fold,
            }
        ]
    if cli.device is not None:
        config["runtime"]["device"] = cli.device
    validate_config(config)

    # Explicit config wins; otherwise preserve scheduler-provided visibility.
    visible_devices = config["runtime"].get("cuda_visible_devices")
    if visible_devices not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    elif os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "") and config["runtime"].get(
        "gpu_ids"
    ):
        visible_devices = ",".join(str(index) for index in config["runtime"]["gpu_ids"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)

    from ssm.runner import train_all

    results = train_all(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
