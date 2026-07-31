import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_reproducible_seed(seed):
    # Seed every RNG in the same order as the original training scripts.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_runtime(tensor, device):
    if device.type == "cuda":
        if device.index is not None:
            return tensor.to(device)
        return tensor.cuda()
    return tensor.to(device)


def mixup_frames(inputs, targets, alpha=0.4):
    # Mix AU frames after flattening the batch and temporal dimensions.
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                )
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        # Validation and checkpoint selection use the shadow parameters.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        # Resume optimization from the non-EMA parameters after evaluation.
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# Lightweight meters preserve the original logging and curve file format.
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = "\t".join(entries)
        print(print_txt)
        if self.log_txt_path:
            with open(self.log_txt_path, "a", encoding="utf-8") as file:
                file.write(print_txt + "\n")

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = (
                correct[:k]
                .contiguous()
                .view(-1)
                .float()
                .sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LegacyCurveRecorder:
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )
        self.epoch_accuracy = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        # Loss is scaled only for sharing the 0-100 plotting axis.
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = "the accuracy/loss curve of train/val"
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])
        y_axis = np.zeros(self.total_epoch)
        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(
            np.arange(0, self.total_epoch + interval_x, interval_x)
        )
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("the training epoch", fontsize=16)
        plt.ylabel("accuracy", fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(
            x_axis,
            y_axis,
            color="g",
            linestyle="-",
            label="train-accuracy",
            lw=2,
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(
            x_axis,
            y_axis,
            color="y",
            linestyle="-",
            label="valid-accuracy",
            lw=2,
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(
            x_axis,
            y_axis,
            color="g",
            linestyle=":",
            label="train-loss-x50",
            lw=2,
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(
            x_axis,
            y_axis,
            color="y",
            linestyle=":",
            label="valid-loss-x50",
            lw=2,
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _unpack_au_targets(au_labels, au_dataset, device):
    # DISFA identity targets are retained for legacy batch compatibility.
    if au_dataset == "disfa":
        au_targets = move_to_runtime(au_labels[0], device)
        identity_targets = move_to_runtime(au_labels[1], device)
    else:
        au_targets = move_to_runtime(au_labels, device)
        identity_targets = None
    return au_targets, identity_targets


def _joint_training_step(
    dfer_batch,
    au_batch,
    model,
    criterion1,
    criterion2,
    optimizer,
    epoch,
    au_dataset,
    au_count,
    device,
    ema,
    moving_losses,
):
    dfer_inputs, dfer_labels = dfer_batch
    au_inputs, au_labels = au_batch

    optimizer.zero_grad()

    dfer_inputs = move_to_runtime(dfer_inputs, device)
    dfer_labels = move_to_runtime(dfer_labels, device)
    au_inputs = move_to_runtime(au_inputs, device)
    au_labels1, au_labels2 = _unpack_au_targets(
        au_labels,
        au_dataset,
        device,
    )

    # Preserve flattened-frame mixup before rebuilding 16-frame clips.
    au_labels1 = au_labels1.reshape(-1, au_count)
    au_inputs = au_inputs.reshape(-1, 3, 224, 224)
    au_inputs, targets_a, targets_b, lam = mixup_frames(
        au_inputs,
        au_labels1,
        alpha=0.4,
    )
    au_inputs = au_inputs.reshape(-1, 16, 3, 224, 224)

    (
        dfer_output,
        dfer_output_pro,
        au_output,
        au_output_pro,
        map_au2emo,
        map_emo2au,
    ) = model(dfer_inputs, au_inputs)

    # Optimize direct and cross-task predictions with four legacy losses.
    loss1 = criterion1(dfer_output, dfer_labels)
    loss2 = lam * criterion2(au_output, targets_a) + (
        1 - lam
    ) * criterion2(au_output, targets_b)
    loss3 = criterion1(dfer_output_pro, dfer_labels)
    loss4 = lam * criterion2(au_output_pro, targets_a) + (
        1 - lam
    ) * criterion2(au_output_pro, targets_b)

    # Normalize each loss by its detached exponential moving average.
    momentum = 0.95
    eps = 1e-8
    if moving_losses[0] is None:
        moving_losses[:] = [
            loss1.detach(),
            loss2.detach(),
            loss3.detach(),
            loss4.detach(),
        ]
    else:
        moving_losses[:] = [
            momentum * moving_losses[0]
            + (1 - momentum) * loss1.detach(),
            momentum * moving_losses[1]
            + (1 - momentum) * loss2.detach(),
            momentum * moving_losses[2]
            + (1 - momentum) * loss3.detach(),
            momentum * moving_losses[3]
            + (1 - momentum) * loss4.detach(),
        ]

    loss1_scaled = loss1 / (moving_losses[0] + eps)
    loss2_scaled = loss2 / (moving_losses[1] + eps)
    loss3_scaled = loss3 / (moving_losses[2] + eps)
    loss4_scaled = loss4 / (moving_losses[3] + eps)

    # Gradually enable the two projected-task losses over 30 epochs.
    p = epoch / 30
    a = 2 / (1 + np.exp(-10 * p)) - 1
    total_loss = (
        loss1_scaled
        + 2.00 * loss2_scaled
        + a * (loss3_scaled + loss4_scaled)
    ) / 3.00

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if ema is not None:
        ema.update()

    return (
        dfer_inputs,
        dfer_labels,
        dfer_output,
        loss1,
        map_au2emo,
        map_emo2au,
    )


def train_joint_epoch(
    train_loader1,
    train_loader2,
    model,
    criterion1,
    criterion2,
    optimizer,
    epoch,
    print_freq,
    log_txt_path,
    au_dataset,
    au_count,
    device,
    ema=None,
):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(
        len(train_loader1),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch),
        log_txt_path=log_txt_path,
    )

    model.train()
    moving_losses = [None, None, None, None]
    len1 = len(train_loader1)
    len2 = len(train_loader2)
    if len1 == 0 or len2 == 0:
        raise RuntimeError(
            "Joint training requires at least one batch from both loaders. "
            f"Got emotion={len1}, AU={len2}."
        )

    # Cycle the shorter loader so every batch from the longer loader is used.
    if len1 >= len2:
        au_iter = iter(train_loader2)
        pair_mode = "emotion"
    else:
        dfer_iter = iter(train_loader1)
        pair_mode = "au"

    if pair_mode == "emotion":
        iterator = enumerate(train_loader1)
        for i, dfer_batch in iterator:
            try:
                au_batch = next(au_iter)
            except StopIteration:
                au_iter = iter(train_loader2)
                au_batch = next(au_iter)

            step = _joint_training_step(
                dfer_batch,
                au_batch,
                model,
                criterion1,
                criterion2,
                optimizer,
                epoch,
                au_dataset,
                au_count,
                device,
                ema,
                moving_losses,
            )
            dfer_inputs, dfer_labels, dfer_output, loss1, maps1, maps2 = step
            acc1_dfer, _ = accuracy(
                dfer_output,
                dfer_labels,
                topk=(1, 5),
            )
            losses.update(loss1.item(), dfer_inputs.size(0))
            top1.update(acc1_dfer[0], dfer_inputs.size(0))
            if i % print_freq == 0:
                progress.display(i)
    else:
        iterator = enumerate(train_loader2)
        for i, au_batch in iterator:
            try:
                dfer_batch = next(dfer_iter)
            except StopIteration:
                dfer_iter = iter(train_loader1)
                dfer_batch = next(dfer_iter)

            step = _joint_training_step(
                dfer_batch,
                au_batch,
                model,
                criterion1,
                criterion2,
                optimizer,
                epoch,
                au_dataset,
                au_count,
                device,
                ema,
                moving_losses,
            )
            dfer_inputs, dfer_labels, dfer_output, loss1, maps1, maps2 = step
            acc1_dfer, _ = accuracy(
                dfer_output,
                dfer_labels,
                topk=(1, 5),
            )
            losses.update(loss1.item(), dfer_inputs.size(0))
            top1.update(acc1_dfer[0], dfer_inputs.size(0))
            if i % print_freq == 0:
                progress.display(i)

    return losses.avg, top1.avg, loss1, maps1, maps2


@torch.no_grad()
def validate_joint(
    val_loader1,
    val_loader2,
    model,
    criterion1,
    print_freq,
    log_txt_path,
    device,
):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(
        len(val_loader1),
        [losses, top1],
        prefix="Test: ",
        log_txt_path=log_txt_path,
    )
    model.eval()

    # Validation intentionally follows the original zip-shortest pairing.
    for i, (
        (dfer_inputs, dfer_labels),
        (au_inputs, au_labels),
    ) in enumerate(zip(val_loader1, val_loader2)):
        dfer_inputs = move_to_runtime(dfer_inputs, device)
        dfer_labels = move_to_runtime(dfer_labels, device)
        au_inputs = move_to_runtime(au_inputs, device)
        dfer_output = model(dfer_inputs, au_inputs)[0]
        loss1 = criterion1(dfer_output, dfer_labels)
        acc1, _ = accuracy(dfer_output, dfer_labels, topk=(1, 5))
        losses.update(loss1.item(), dfer_inputs.size(0))
        top1.update(acc1[0], dfer_inputs.size(0))
        if i % print_freq == 0:
            progress.display(i)

    if top1.count == 0:
        raise RuntimeError(
            "Joint validation received no paired emotion/AU batches."
        )

    print("Current Accuracy: {top1.avg:.3f}".format(top1=top1))
    with open(log_txt_path, "a", encoding="utf-8") as file:
        file.write(
            "Current Accuracy: {top1.avg:.3f}".format(top1=top1)
            + "\n"
        )
    return top1.avg, losses.avg


def build_legacy_optimizer(model):
    # Mapping deltas, CLIP encoder, and remaining layers use separate rates.
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-6},
            {"params": [model.delta_map1], "lr": 1e-2},
            {"params": [model.delta_map2], "lr": 1e-2},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if not name.startswith("encoder")
                    and not name.startswith("delta_map1")
                    and not name.startswith("delta_map2")
                ],
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )
