# Verify that checkpoints serialized by the original scripts remain loadable.
import __main__
from pathlib import Path

import torch

from ssm.runner import load_checkpoint_state


def test_legacy_main_recorder_checkpoint_loads(tmp_path):
    recorder_type = type("RecorderMeter", (), {})
    recorder_type.__module__ = "__main__"
    previous = getattr(__main__, "RecorderMeter", None)
    existed = hasattr(__main__, "RecorderMeter")
    __main__.RecorderMeter = recorder_type

    source = torch.nn.Linear(3, 2)
    recorder = recorder_type()
    recorder.total_epoch = 30
    checkpoint_path = Path(tmp_path) / "legacy.pth"
    try:
        torch.save(
            {
                "state_dict": source.state_dict(),
                "recorder": recorder,
            },
            checkpoint_path,
        )
    finally:
        if existed:
            __main__.RecorderMeter = previous
        else:
            delattr(__main__, "RecorderMeter")

    target = torch.nn.Linear(3, 2)
    loaded = load_checkpoint_state(
        target,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert loaded["recorder"].total_epoch == 30
    for name, expected in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], expected)
