# Exercise failure-path guards without changing successful numerical paths.
import inspect

import pytest
import torch

from ssm.evaluation.metrics import (
    evaluate_action_units,
    evaluate_emotions,
)
from ssm.runner import evaluate_checkpoint, train_fold_pair
from ssm.training.legacy import (
    move_to_runtime,
    train_joint_epoch,
    validate_joint,
)


class _Model:
    def train(self):
        return None

    def eval(self):
        return None


class _MoveRecorder:
    def __init__(self):
        self.calls = []

    def cuda(self):
        self.calls.append(("cuda", None))
        return self

    def to(self, device):
        self.calls.append(("to", device))
        return self


def test_empty_paired_loaders_fail_before_metrics_or_training():
    model = _Model()
    device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="at least one batch"):
        train_joint_epoch(
            [],
            [],
            model,
            criterion1=None,
            criterion2=None,
            optimizer=None,
            epoch=0,
            print_freq=100,
            log_txt_path="",
            au_dataset="bp4d",
            au_count=12,
            device=device,
        )

    with pytest.raises(RuntimeError, match="no paired"):
        validate_joint(
            [],
            [],
            model,
            criterion1=None,
            print_freq=100,
            log_txt_path="",
            device=device,
        )

    with pytest.raises(RuntimeError, match="complete ordered"):
        evaluate_action_units(
            [],
            [],
            model,
            device,
            "bp4d",
            12,
        )

    with pytest.raises(RuntimeError, match="no paired"):
        evaluate_emotions(
            [],
            [],
            model,
            device,
            ["happiness"] * 7,
            "DFEW",
            1,
        )


def test_explicit_cuda_index_is_not_replaced_by_default_cuda():
    tensor = _MoveRecorder()
    device = torch.device("cuda:1")
    assert move_to_runtime(tensor, device) is tensor
    assert tensor.calls == [("to", device)]

    tensor = _MoveRecorder()
    assert move_to_runtime(tensor, torch.device("cuda")) is tensor
    assert tensor.calls == [("cuda", None)]


def test_first_epoch_always_initializes_best_checkpoints():
    source = "".join(inspect.getsource(train_fold_pair).split())
    assert "is_best_acc=epoch==0orval_acc>best_acc" in source
    assert "is_best=epoch==0orval_uar>best_uar" in source
    assert 'is_best,paths["checkpoint"],paths["best_emotion"]' in source
    assert '"best_uar":best_uar' in source
    assert "ifepoch==0orf1_mean>max_f1:" in source
    assert "ifepoch==0orauc_mean>max_auc:" in source


def test_checkpoint_evaluation_uses_complete_emotion_validation():
    source = "".join(inspect.getsource(evaluate_checkpoint).split())
    assert (
        'complete_emotion_evaluation=(config["evaluation"].get('
        '"legacy_validation_pairing")=="complete_emotion")' in source
    )
    assert (
        "au_evaluation_emotion_loader=getattr(val_loader1,"
        '"au_companion_loader",val_loader1,)' in source
    )
    assert "evaluate_action_units(au_evaluation_emotion_loader," in source
    assert "complete_primary=complete_emotion_evaluation" in source
