import itertools
import os
from collections import OrderedDict

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import SequentialSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssm.training.legacy import (
    _pair_primary_with_restarted_secondary,
    move_to_runtime,
)
from .adaptive_vote import search_segment_safe_vote


def majority_voting(preds, K=3, p=0.5):
    # Edge padding preserves the original temporal voting behavior.
    T, A = preds.shape
    window_size = 2 * K + 1
    padded = np.pad(preds, ((K, K), (0, 0)), mode="edge")
    final = np.zeros_like(preds)
    for t in range(T):
        window = padded[t : t + window_size]
        counts = window.sum(axis=0)
        final[t] = (counts >= p * window_size).astype(int)
    return final


def _inclusive_float_grid(start, stop, step):
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not (0.0 < start <= stop <= 1.0) or step <= 0.0:
        raise ValueError("Invalid adaptive AU threshold range.")
    count = int(round((stop - start) / step))
    if not np.isclose(start + count * step, stop):
        raise ValueError("AU threshold range must include its stop value.")
    return tuple(round(start + index * step, 12) for index in range(count + 1))


def _complete_p_candidates(K_values):
    return tuple(
        sorted(
            {
                required / float(2 * K + 1)
                for K in K_values
                for required in range(1, 2 * K + 2)
            }
        )
    )


@torch.no_grad()
def evaluate_action_units(
    val_loader1,
    val_loader2,
    model,
    device,
    au_dataset,
    au_count,
    evaluation_config=None,
):
    evaluation_config = evaluation_config or {}
    adaptive = evaluation_config.get("adaptive_vote", {})
    threshold_config = evaluation_config.get("au_threshold", {})
    complete_loader = getattr(val_loader2, "complete_au_loader", None)
    if complete_loader is None:
        raise RuntimeError("The complete ordered AU evaluation loader is unavailable.")
    if bool(complete_loader.drop_last):
        raise RuntimeError("Complete AU evaluation requires drop_last=False.")
    if not isinstance(complete_loader.sampler, SequentialSampler):
        raise RuntimeError("Complete AU evaluation requires ordered samples.")
    dataset = complete_loader.dataset
    required_metadata = (
        "get_eval_metadata",
        "track_frame_count",
        "segment_count_with_targets",
    )
    if any(not hasattr(dataset, name) for name in required_metadata):
        raise TypeError("Complete AU dataset metadata is unavailable.")

    model.eval()
    dfer_iterator = iter(val_loader1)
    segment_records = OrderedDict()
    seen_frames = set()
    loss_sum = 0.0
    loss_elements = 0
    evaluated_frames = 0
    sample_cursor = 0

    for au_inputs, au_labels in complete_loader:
        try:
            dfer_inputs, _ = next(dfer_iterator)
        except StopIteration:
            dfer_iterator = iter(val_loader1)
            try:
                dfer_inputs, _ = next(dfer_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "AU evaluation received no compatible emotion batches."
                ) from error
        dfer_inputs = move_to_runtime(dfer_inputs, device)
        au_inputs = move_to_runtime(au_inputs, device)
        if au_dataset == "disfa":
            target = move_to_runtime(au_labels[0], device)
        else:
            target = move_to_runtime(au_labels, device)
        output = model(dfer_inputs, au_inputs)[2]
        batch_size = int(au_inputs.shape[0])
        if target.ndim != 3 or target.shape[0] != batch_size:
            raise RuntimeError("Complete AU targets must have shape [B, T, A].")
        clip_len = int(target.shape[1])
        target = target.reshape(batch_size, clip_len, int(au_count))
        expected_output_shape = (
            batch_size * clip_len,
            int(au_count),
        )
        if tuple(output.shape) != expected_output_shape:
            raise RuntimeError(
                "AU output/target size mismatch: "
                f"output {tuple(output.size())}, expected "
                f"{expected_output_shape}."
            )
        output = output.reshape_as(target)
        probabilities = torch.sigmoid(output).detach().cpu()
        targets_cpu = target.detach().cpu()
        element_loss = (
            F.binary_cross_entropy_with_logits(
                output,
                target,
                reduction="none",
            )
            .detach()
            .cpu()
        )

        for row in range(batch_size):
            metadata = dataset.get_eval_metadata(sample_cursor + row)
            target_mask = metadata["target_mask"]
            frame_indices = metadata["frame_indices"]
            if len(target_mask) != clip_len or len(frame_indices) != clip_len:
                raise RuntimeError("AU evaluation metadata length mismatch.")
            for column, valid in enumerate(target_mask):
                if not valid:
                    continue
                records = segment_records.setdefault(
                    metadata["segment_id"],
                    {},
                )
                frame_index = int(frame_indices[column])
                frame_key = (metadata["segment_id"], frame_index)
                if frame_key in seen_frames:
                    raise RuntimeError(
                        "Duplicate AU evaluation frame: " + repr(frame_key)
                    )
                seen_frames.add(frame_key)
                records[frame_index] = (
                    targets_cpu[row, column].numpy(),
                    probabilities[row, column].numpy(),
                )
                loss_sum += float(element_loss[row, column].sum().item())
                loss_elements += int(au_count)
                evaluated_frames += 1
        sample_cursor += batch_size

    if sample_cursor == 0:
        raise RuntimeError("AU evaluation received no compatible paired batches.")
    if sample_cursor != len(dataset):
        raise RuntimeError(
            f"Incomplete AU samples: evaluated={sample_cursor}, "
            f"expected={len(dataset)}."
        )
    if evaluated_frames != int(dataset.track_frame_count):
        raise RuntimeError(
            f"Incomplete AU frames: evaluated={evaluated_frames}, "
            f"expected={dataset.track_frame_count}."
        )
    if len(segment_records) != int(dataset.segment_count_with_targets):
        raise RuntimeError("Incomplete AU continuous-segment reconstruction.")

    segments = []
    for segment_id, records in segment_records.items():
        frame_indices = sorted(records)
        expected = list(range(frame_indices[0], frame_indices[-1] + 1))
        if frame_indices != expected:
            raise RuntimeError("Non-contiguous AU segment: " + segment_id)
        ordered = [records[index] for index in frame_indices]
        segments.append(
            (
                np.stack([item[0] for item in ordered], axis=0),
                np.stack([item[1] for item in ordered], axis=0),
            )
        )

    thresholds = _inclusive_float_grid(
        threshold_config.get("search_start", 0.01),
        threshold_config.get("search_stop", 0.99),
        threshold_config.get("search_step", 0.01),
    )
    K_values = tuple(
        range(
            int(adaptive.get("half_window_min", 0)),
            int(adaptive.get("half_window_max", 7)) + 1,
        )
    )
    tie = adaptive.get("tie_reference", {})
    tie_reference = (
        int(tie.get("half_window", 3)),
        float(tie.get("positive_ratio", 0.3)),
        float(tie.get("threshold", 0.5)),
    )
    result = search_segment_safe_vote(
        segments,
        thresholds=thresholds,
        K_values=K_values,
        p_values=_complete_p_candidates(K_values),
        expected_au_count=int(au_count),
        tie_reference=tie_reference,
    )
    all_targets = np.concatenate([item[0] for item in segments], axis=0)
    all_probabilities = np.concatenate(
        [item[1] for item in segments],
        axis=0,
    )
    auc_scores = []
    for au_index in range(int(au_count)):
        try:
            auc_scores.append(
                float(
                    roc_auc_score(
                        all_targets[:, au_index],
                        all_probabilities[:, au_index],
                    )
                )
            )
        except ValueError as error:
            print(f"Error computing AUC for AU{au_index + 1}: {error}")
            auc_scores.append(float("nan"))
    mean_auc = float(np.nanmean(auc_scores))
    mean_loss = float(loss_sum / loss_elements)
    result.update(
        {
            "dataset": str(au_dataset).lower(),
            "selection_scope": "current_evaluation_labels",
            "loss": mean_loss,
            "auc": mean_auc,
            "per_au_auc": auc_scores,
            "segments": int(len(segments)),
        }
    )
    print(
        f"{str(au_dataset).upper()} adaptive AU metric: "
        f"frames={result['evaluated_frames']}, "
        f"F1={result['f1']:.4f}, AUC={mean_auc:.4f}, "
        f"threshold={result['threshold']:.2f}, "
        f"K={result['K']}, p={result['p']:.6g}, "
        f"votes={result['required_votes']}/{result['window_size']}"
    )
    print("Adaptive per-AU F1:", result["per_au_f1"])
    return result, float(result["f1"]), mean_auc


def plot_confusion_matrix(
    matrix,
    classes,
    normalize=False,
    title="confusion matrix",
    cmap=plt.cm.Blues,
):
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.0
    for i, j in itertools.product(
        range(matrix.shape[0]),
        range(matrix.shape[1]),
    ):
        plt.text(
            j,
            i,
            format(matrix[i, j], fmt),
            fontsize=12,
            horizontalalignment="center",
            color="white" if matrix[i, j] > thresh else "black",
        )
    plt.ylabel("True label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.tight_layout()


@torch.no_grad()
def evaluate_emotions(
    val_loader1,
    val_loader2,
    model,
    device,
    class_names,
    dataset_name,
    emotion_fold,
    confusion_matrix_path=None,
    log_txt_path=None,
    complete_primary=False,
):
    model.eval()
    correct = 0
    evaluated_batches = 0
    evaluated_samples = 0
    predicted_batches = []
    target_batches = []

    paired_batches = (
        _pair_primary_with_restarted_secondary(
            val_loader1,
            val_loader2,
        )
        if complete_primary
        else zip(val_loader1, val_loader2)
    )
    for i, (
        (dfer_inputs, dfer_labels),
        (au_inputs, au_labels),
    ) in enumerate(paired_batches):
        dfer_inputs = move_to_runtime(dfer_inputs, device)
        target = move_to_runtime(dfer_labels, device)
        au_inputs = move_to_runtime(au_inputs, device)
        output = model(dfer_inputs, au_inputs)[0]
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        evaluated_batches += 1
        evaluated_samples += int(target.numel())
        predicted_batches.append(predicted.detach().cpu())
        target_batches.append(target.detach().cpu())

    if evaluated_batches == 0:
        raise RuntimeError("Emotion evaluation received no paired emotion/AU batches.")

    all_predicted = torch.cat(predicted_batches, 0)
    all_targets = torch.cat(target_batches, 0)
    war = 100.0 * correct / len(val_loader1.dataset)
    if complete_primary:
        war = 100.0 * correct / evaluated_samples
    targets_numpy = all_targets.data.cpu().numpy().reshape(-1)
    predicted_numpy = all_predicted.cpu().numpy().reshape(-1)
    class_count = len(class_names)
    if (
        np.any(targets_numpy < 0)
        or np.any(targets_numpy >= class_count)
        or np.any(predicted_numpy < 0)
        or np.any(predicted_numpy >= class_count)
    ):
        raise RuntimeError("Emotion prediction is outside the class range.")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (targets_numpy, predicted_numpy), 1)
    np.set_printoptions(precision=4)
    support = matrix.sum(axis=1)
    normalized_cm = np.divide(
        matrix.astype("float"),
        support[:, np.newaxis],
        out=np.zeros_like(matrix, dtype=float),
        where=support[:, np.newaxis] != 0,
    )
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag[support > 0].mean()

    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    if confusion_matrix_path is not None:
        plt.figure(figsize=(10, 8))
        if dataset_name == "FERV39K":
            title = "Confusion Matrix on FERV39k"
        else:
            title = f"Confusion Matrix on {dataset_name} fold {emotion_fold}"
        plot_confusion_matrix(
            normalized_cm,
            classes=class_names,
            normalize=True,
            title=title,
        )
        plt.savefig(os.path.join(confusion_matrix_path))
        plt.close()

    if log_txt_path is not None:
        with open(log_txt_path, "a", encoding="utf-8") as file:
            file.write("************************\n")
            file.write("Confusion Matrix Diag:\n")
            file.write(str(list_diag.tolist()) + "\n")
            file.write("UAR: {:.2f}".format(uar) + "\n")
            file.write("WAR: {:.2f}".format(war) + "\n")
            file.write("************************\n")

    return uar, war
