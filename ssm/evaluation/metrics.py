import itertools
import os

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssm.training.legacy import move_to_runtime


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


@torch.no_grad()
def evaluate_action_units(
    val_loader1,
    val_loader2,
    model,
    device,
    au_dataset,
    au_count,
    K=3,
    p=0.3,
):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    all_preds = []
    all_targets = []
    loss_total = 0.0
    loss_count = 0

    # Both tasks are forwarded together using zip-shortest validation.
    for (
        (dfer_inputs, dfer_labels),
        (au_inputs, au_labels),
    ) in zip(val_loader1, val_loader2):
        dfer_inputs = move_to_runtime(dfer_inputs, device)
        dfer_labels = move_to_runtime(dfer_labels, device)
        au_inputs = move_to_runtime(au_inputs, device)
        if au_dataset == "disfa":
            target = move_to_runtime(au_labels[0], device)
            _identity_labels = move_to_runtime(au_labels[1], device)
        else:
            target = move_to_runtime(au_labels, device)
            _identity_labels = None

        output = model(dfer_inputs, au_inputs)[2]
        target = target.reshape(-1, au_count)
        if output.size() != target.size():
            raise RuntimeError(
                "AU output/target size mismatch: "
                f"output {tuple(output.size())}, "
                f"target {tuple(target.size())}."
            )

        loss = criterion(output, target)
        loss_total += loss.item()
        loss_count += 1
        probs = torch.sigmoid(output)
        all_preds.append(probs.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    if loss_count == 0:
        raise RuntimeError(
            "AU evaluation received no compatible paired batches."
        )

    mean_loss = loss_total / loss_count
    print("* loss {:.3f}".format(mean_loss))

    y_probs = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    threshold = 0.5
    y_pred = (y_probs >= threshold).astype(int)
    y_post = majority_voting(y_pred, K=K, p=p)

    A = y_true.shape[1]
    f1_raw = [
        f1_score(y_true[:, i], y_pred[:, i])
        for i in range(A)
    ]
    f1_post = [
        f1_score(y_true[:, i], y_post[:, i])
        for i in range(A)
    ]
    print(
        f"F1 raw  mean: {np.mean(f1_raw):.4f}, per-AU: {f1_raw}"
    )
    print(
        f"F1 post mean: {np.mean(f1_post):.4f}, per-AU: {f1_post}"
    )

    # Select one global threshold by mean post-processed F1 across AUs.
    f1_score_ls = []
    for i in range(1, 100):
        th = i * 0.01
        y_thr = (y_probs >= th).astype(int)
        y_thr_post = majority_voting(y_thr, K=K, p=p)
        f1_scores = [
            f1_score(y_true[:, j], y_thr_post[:, j])
            for j in range(A)
        ]
        f1_score_ls.append(f1_scores)
    f1_score_arr = np.array(f1_score_ls)
    max_idx = np.argmax(np.mean(f1_score_arr, axis=1))
    best_row = f1_score_arr[max_idx]
    print(
        "Best post-processed F1 mean: "
        f"{best_row.mean():.4f} at threshold {(max_idx + 1) / 100}, "
        f"scores: {best_row}"
    )

    # AUC is computed from raw sigmoid probabilities without voting.
    auc_scores = []
    for i in range(A):
        try:
            auc_scores.append(
                roc_auc_score(y_true[:, i], y_probs[:, i])
            )
        except ValueError as error:
            print(f"Error computing AUC for AU{i + 1}: {error}")
    mean_auc = np.mean(auc_scores)
    print(f"AUC mean: {mean_auc:.4f}, per-AU: {auc_scores}")

    return {"loss": mean_loss}, best_row.mean(), mean_auc


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
):
    model.eval()
    correct = 0
    evaluated_batches = 0

    # Expression metrics use the same paired forward path as AU metrics.
    for i, (
        (dfer_inputs, dfer_labels),
        (au_inputs, au_labels),
    ) in enumerate(zip(val_loader1, val_loader2)):
        dfer_inputs = move_to_runtime(dfer_inputs, device)
        target = move_to_runtime(dfer_labels, device)
        au_inputs = move_to_runtime(au_inputs, device)
        output = model(dfer_inputs, au_inputs)[0]
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        evaluated_batches += 1

        if i == 0:
            all_predicted = predicted
            all_targets = target
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, target), 0)

    if evaluated_batches == 0:
        raise RuntimeError(
            "Emotion evaluation received no paired emotion/AU batches."
        )

    # Keep the full expression-dataset denominator used in the release runs.
    war = 100.0 * correct / len(val_loader1.dataset)
    matrix = confusion_matrix(
        all_targets.data.cpu().numpy(),
        all_predicted.cpu().numpy(),
    )
    np.set_printoptions(precision=4)
    normalized_cm = (
        matrix.astype("float")
        / matrix.sum(axis=1)[:, np.newaxis]
    )
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()

    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    if confusion_matrix_path is not None:
        plt.figure(figsize=(10, 8))
        if dataset_name == "FERV39K":
            title = "Confusion Matrix on FERV39k"
        else:
            title = (
                f"Confusion Matrix on {dataset_name} "
                f"fold {emotion_fold}"
            )
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
