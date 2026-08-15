# Guard BP4D sampling, augmentation, optimization, and evaluation parity.
import ast
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ssm.data.emotion import EmotionVideoDataset
from ssm.data.group_transforms import (
    ColorJitter,
    GroupRandomHorizontalFlip,
    GroupRandomSizedCrop,
)
from ssm.training import legacy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_TRAINING_PATH = REPOSITORY_ROOT / "ssm" / "training" / "legacy.py"
RUNNER_PATH = REPOSITORY_ROOT / "ssm" / "runner.py"
EMOTION_DATA_PATH = REPOSITORY_ROOT / "ssm" / "data" / "emotion.py"
METRICS_PATH = REPOSITORY_ROOT / "ssm" / "evaluation" / "metrics.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(path, name):
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    raise AssertionError(f"Function {name!r} was not found in {path}.")


def _call_name(call):
    node = call.func
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def _call_lines(function, name):
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def _compact_function_source(path, name):
    text = path.read_text(encoding="utf-8")
    node = _function(path, name)
    source = ast.get_source_segment(text, node)
    return "".join(source.split())


def _dataset_branch(function, dataset):
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == dataset
        ):
            continue
        return node.body
    raise AssertionError(f"Dataset branch {dataset!r} was not found.")


def _branch_calls(branch, name):
    calls = []
    for statement in branch:
        calls.extend(
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and _call_name(node) == name
        )
    return calls


def test_mixup_uses_one_partner_for_every_frame_in_a_clip():
    frames = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3, 1, 1)
    targets = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)

    np.random.seed(2026)
    torch.manual_seed(490)
    expected_lam = np.random.beta(0.4, 0.4)
    frame_permutation = torch.randperm(frames.size(0) * frames.size(1))
    expected_order = []
    for flat_index in frame_permutation.tolist():
        clip_index = flat_index // frames.size(1)
        if clip_index not in expected_order:
            expected_order.append(clip_index)
    expected_index = torch.tensor(expected_order)
    expected_frames = (
        expected_lam * frames + (1 - expected_lam) * frames[expected_index, :]
    )
    expected_targets_b = targets[expected_index]
    expected_next_numpy = np.random.random()
    expected_next_torch = torch.rand(4)

    np.random.seed(2026)
    torch.manual_seed(490)
    mixed, targets_a, targets_b, lam = legacy.mixup_clips(
        frames,
        targets,
        alpha=0.4,
    )

    assert lam == expected_lam
    assert torch.equal(mixed, expected_frames)
    assert targets_a is targets
    assert torch.equal(targets_b, expected_targets_b)
    assert np.random.random() == expected_next_numpy
    assert torch.equal(torch.rand(4), expected_next_torch)


def test_ema_update_apply_and_restore_match_the_legacy_equations():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema = legacy.ModelEMA(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(6.0)

    expected_shadow = torch.tensor([[2.0]]) * 0.99 + torch.tensor([[6.0]]) * 0.01
    ema.update()
    assert torch.equal(ema.shadow["weight"], expected_shadow)

    ema.apply_shadow()
    assert torch.equal(model.weight, expected_shadow)
    ema.restore()
    assert torch.equal(model.weight, torch.tensor([[6.0]]))
    assert ema.backup == {}


def test_emotion_segment_sampling_matches_the_legacy_branches_and_rng():
    class Record:
        def __init__(self, num_frames):
            self.num_frames = num_frames

    dataset = object.__new__(EmotionVideoDataset)
    dataset.duration = 1
    dataset.num_segments = 16

    for num_frames in (1, 8, 16, 17, 32, 320):
        record = Record(num_frames)
        for seed in (0, 1, 31, 777):
            np.random.seed(seed)
            actual = dataset._get_train_indices(record)
            actual_next = np.random.randint(0, 2**20)

            np.random.seed(seed)
            average_duration = (
                num_frames - dataset.duration + 1
            ) // dataset.num_segments
            if average_duration > 0:
                expected = np.multiply(
                    list(range(dataset.num_segments)),
                    average_duration,
                ) + np.random.randint(
                    average_duration,
                    size=dataset.num_segments,
                )
            elif num_frames > dataset.num_segments:
                expected = np.sort(
                    np.random.randint(
                        num_frames - dataset.duration + 1,
                        size=dataset.num_segments,
                    )
                )
            else:
                expected = np.pad(
                    np.array(list(range(num_frames))),
                    (0, dataset.num_segments - num_frames),
                    "edge",
                )
            expected_next = np.random.randint(0, 2**20)

            assert np.array_equal(actual, expected)
            assert actual_next == expected_next

        if num_frames > dataset.num_segments + dataset.duration - 1:
            tick = (num_frames - dataset.duration + 1) / float(dataset.num_segments)
            expected_test = np.array(
                [
                    int(tick / 2.0 + tick * index)
                    for index in range(dataset.num_segments)
                ]
            )
        else:
            expected_test = np.pad(
                np.array(list(range(num_frames))),
                (0, dataset.num_segments - num_frames),
                "edge",
            )
        assert np.array_equal(
            dataset._get_test_indices(record),
            expected_test,
        )


def test_active_group_augmentations_preserve_rng_call_order():
    pixels = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(
        240,
        320,
        3,
    )
    images = [Image.fromarray(np.roll(pixels, shift, axis=1)) for shift in (0, 7)]

    seed = 19
    reference_rng = random.Random(seed)
    area = images[0].size[0] * images[0].size[1]
    found = False
    for _ in range(10):
        target_area = reference_rng.uniform(0.08, 1.0) * area
        aspect_ratio = reference_rng.uniform(3.0 / 4, 4.0 / 3)
        width = int(round((target_area * aspect_ratio) ** 0.5))
        height = int(round((target_area / aspect_ratio) ** 0.5))
        if reference_rng.random() < 0.5:
            width, height = height, width
        if width <= images[0].size[0] and height <= images[0].size[1]:
            x_offset = reference_rng.randint(
                0,
                images[0].size[0] - width,
            )
            y_offset = reference_rng.randint(
                0,
                images[0].size[1] - height,
            )
            found = True
            break
    assert found

    expected_crops = [
        image.crop(
            (
                x_offset,
                y_offset,
                x_offset + width,
                y_offset + height,
            )
        ).resize((64, 64), Image.BILINEAR)
        for image in images
    ]
    expected_next = reference_rng.random()

    random.seed(seed)
    actual_crops = GroupRandomSizedCrop(64)([image.copy() for image in images])
    actual_next = random.random()
    for actual, expected in zip(actual_crops, expected_crops):
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert actual_next == expected_next

    reference_rng = random.Random(seed)
    should_flip = reference_rng.random() < 0.5
    expected_next = reference_rng.random()
    random.seed(seed)
    flipped = GroupRandomHorizontalFlip()([image.copy() for image in images])
    actual_next = random.random()
    if should_flip:
        expected_flips = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
    else:
        expected_flips = images
    for actual, expected in zip(flipped, expected_flips):
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert actual_next == expected_next

    reference_rng = random.Random(seed)
    brightness = reference_rng.uniform(0.5, 1.5)
    expected_next = reference_rng.random()
    random.seed(seed)
    jittered = ColorJitter(brightness=0.5)([image.copy() for image in images])
    actual_next = random.random()
    from torchvision.transforms.functional import adjust_brightness

    expected_jittered = [adjust_brightness(image, brightness) for image in images]
    for actual, expected in zip(jittered, expected_jittered):
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert actual_next == expected_next


def test_emotion_training_transform_order_remains_legacy_compatible():
    source = _compact_function_source(
        EMOTION_DATA_PATH,
        "build_emotion_training_dataset",
    )
    ordered = [
        "ColorJitter(brightness=0.5)",
        "GroupRandomSizedCrop(image_size)",
        "GroupRandomHorizontalFlip()",
        "Stack()",
        "ToTorchFormatTensor()",
        "GroupNormalize(mean=CLIP_MEAN,std=CLIP_STD)",
    ]
    positions = [source.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)


def test_ferv39k_training_transform_order_remains_compatible():
    source = _compact_function_source(
        EMOTION_DATA_PATH,
        "build_emotion_training_dataset",
    )
    branch = source[source.index('elifargs.dataset=="FERV39K"') :]
    branch = branch[: branch.index("else:")]
    ordered = [
        "RandomRotation(4)",
        "GroupRandomSizedCrop(image_size)",
        "GroupRandomHorizontalFlip()",
        "Stack()",
        "ToTorchFormatTensor()",
        "GroupNormalize(mean=CLIP_MEAN,std=CLIP_STD)",
    ]
    positions = [branch.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)


def test_mafw_training_and_evaluation_normalization_remain_compatible():
    training_function = _function(
        EMOTION_DATA_PATH,
        "build_emotion_training_dataset",
    )
    branch = _dataset_branch(training_function, "MAFW")
    ordered_names = [
        "GroupRandomSizedCrop",
        "GroupRandomHorizontalFlip",
        "Stack",
        "ToTorchFormatTensor",
        "GroupNormalize",
    ]
    calls = {name: _branch_calls(branch, name) for name in ordered_names}
    assert all(len(items) == 1 for items in calls.values())
    positions = [calls[name][0].lineno for name in ordered_names]
    assert positions == sorted(positions)

    normalize = calls["GroupNormalize"][0]
    assert [keyword.arg for keyword in normalize.keywords] == [
        "mean",
        "std",
    ]
    assert [
        keyword.value.id
        for keyword in normalize.keywords
        if isinstance(keyword.value, ast.Name)
    ] == ["CLIP_MEAN", "CLIP_STD"]

    evaluation = _compact_function_source(
        EMOTION_DATA_PATH,
        "build_emotion_evaluation_dataset",
    )
    assert "GroupResize(image_size)" in evaluation
    assert "GroupNormalize(mean=CLIP_MEAN,std=CLIP_STD,)" in evaluation


def test_bp4d_training_iterator_starts_au_first_and_restarts_it():
    trace = []

    class TraceLoader:
        def __init__(self, name, length):
            self.name = name
            self.items = [object() for _ in range(length)]

        def __len__(self):
            return len(self.items)

        def __iter__(self):
            trace.append(self.name)
            return iter(self.items)

    class Model:
        def train(self):
            return None

    def fake_step(*args, **kwargs):
        inputs = torch.zeros(1, 1)
        labels = torch.zeros(1, dtype=torch.long)
        logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        loss = torch.tensor(1.0)
        mapping = torch.zeros(1, 1)
        return inputs, labels, logits, loss, mapping, mapping

    emotion_loader = TraceLoader("emotion", 3)
    au_loader = TraceLoader("au", 1)

    with patch.object(legacy, "_joint_training_step", fake_step):
        legacy.train_joint_epoch(
            emotion_loader,
            au_loader,
            Model(),
            criterion1=None,
            criterion2=None,
            optimizer=None,
            epoch=0,
            print_freq=100,
            log_txt_path="",
            au_dataset="bp4d",
            au_count=12,
            device=torch.device("cpu"),
            ema=None,
        )

    assert trace[:2] == ["au", "emotion"]
    assert trace == ["au", "emotion", "au", "au"]


def test_joint_step_locks_clip_mixup():
    source = _compact_function_source(
        LEGACY_TRAINING_PATH,
        "_joint_training_step",
    )
    required = [
        "mixup_clips(au_inputs,au_labels1,alpha=mixup_alpha,)",
        "targets_a_flat=targets_a.reshape(-1,au_count)",
        "targets_b_flat=targets_b.reshape(-1,au_count)",
        "loss1=criterion1(dfer_output,dfer_labels)",
        "loss2=lam*criterion2(au_output,targets_a_flat)+(1-lam)"
        "*criterion2(au_output,targets_b_flat)",
        "loss3=criterion1(dfer_output_pro,dfer_labels)",
        "loss4=lam*criterion2(au_output_pro,targets_a_flat)+(1-lam)"
        "*criterion2(au_output_pro,targets_b_flat)",
        "momentum=0.95",
        "eps=1e-8",
        "p=epoch/30",
        "a=2/(1+np.exp(-10*p))-1",
        "loss1_scaled+2.00*loss2_scaled+a*(loss3_scaled+loss4_scaled)",
        ")/3.00",
        "total_loss.backward()",
        "torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)",
        "optimizer.step()",
        "ema.update()",
    ]
    for fragment in required:
        assert fragment in source

    positions = [
        source.index("optimizer.zero_grad()"),
        source.index("mixup_clips("),
        source.index("=model(dfer_inputs,au_inputs)"),
        source.index("total_loss.backward()"),
        source.index("clip_grad_norm_("),
        source.index("optimizer.step()"),
        source.index("ema.update()"),
    ]
    assert positions == sorted(positions)


def test_runner_preserves_scheduler_ema_and_evaluation_order():
    function = _function(RUNNER_PATH, "train_fold_pair")
    calls = {
        "train": _call_lines(function, "train_joint_epoch")[0],
        "scheduler": _call_lines(function, "scheduler.step")[0],
        "apply_ema": _call_lines(function, "ema.apply_shadow")[0],
        "validate": _call_lines(function, "validate_joint")[0],
        "au_eval": _call_lines(function, "evaluate_action_units")[0],
        "restore_ema": _call_lines(function, "ema.restore")[0],
        "emotion_eval": _call_lines(function, "evaluate_emotions")[0],
    }
    assert list(calls.values()) == sorted(calls.values())


def test_dfer_validation_and_evaluation_zip_the_two_loaders():
    checks = [
        (LEGACY_TRAINING_PATH, "validate_joint"),
        (METRICS_PATH, "evaluate_emotions"),
    ]
    for path, function_name in checks:
        function = _function(path, function_name)
        zip_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _call_name(node) == "zip"
        ]
        assert len(zip_calls) == 1
        assert [
            argument.id
            for argument in zip_calls[0].args
            if isinstance(argument, ast.Name)
        ] == ["val_loader1", "val_loader2"]

    au_source = _compact_function_source(
        METRICS_PATH,
        "evaluate_action_units",
    )
    assert 'complete_loader=getattr(val_loader2,"complete_au_loader",None)' in au_source
    assert "dfer_iterator=iter(val_loader1)" in au_source


def test_adaptive_vote_search_and_war_denominator():
    metrics_tree = _tree(METRICS_PATH)
    vote_node = next(
        node
        for node in metrics_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "majority_voting"
    )
    vote_module = ast.fix_missing_locations(
        ast.Module(body=[vote_node], type_ignores=[])
    )
    namespace = {"np": np}
    exec(compile(vote_module, str(METRICS_PATH), "exec"), namespace)

    predictions = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int64,
    )
    expected = np.zeros_like(predictions)
    padded = np.pad(predictions, ((1, 1), (0, 0)), mode="edge")
    for index in range(len(predictions)):
        counts = padded[index : index + 3].sum(axis=0)
        expected[index] = (counts >= 0.5 * 3).astype(int)
    actual = namespace["majority_voting"](predictions, K=1, p=0.5)
    assert np.array_equal(actual, expected)

    au_source = _compact_function_source(
        METRICS_PATH,
        "evaluate_action_units",
    )
    assert "search_segment_safe_vote(" in au_source
    assert "thresholds=thresholds" in au_source
    assert "K_values=K_values" in au_source
    assert "p_values=_complete_p_candidates(K_values)" in au_source
    assert "expected_au_count=int(au_count)" in au_source

    emotion_function = _function(METRICS_PATH, "evaluate_emotions")
    war_assignment = next(
        node
        for node in ast.walk(emotion_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "war"
            for target in node.targets
        )
    )
    assert isinstance(war_assignment.value, ast.BinOp)
    assert isinstance(war_assignment.value.op, ast.Div)
    denominator = war_assignment.value.right
    assert isinstance(denominator, ast.Call)
    assert _call_name(denominator) == "len"
    assert len(denominator.args) == 1
    dataset = denominator.args[0]
    assert isinstance(dataset, ast.Attribute)
    assert dataset.attr == "dataset"
    assert isinstance(dataset.value, ast.Name)
    assert dataset.value.id == "val_loader1"


def test_runner_builds_emotion_loaders_before_bp4d_loaders():
    function = _function(RUNNER_PATH, "build_data_loaders")
    assignments = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node.lineno)

    order = [
        min(assignments["train_data1"]),
        min(assignments["test_data1"]),
        min(assignments["train_loader1"]),
        min(assignments["val_loader1"]),
        min(assignments["train_data2"]),
        min(assignments["test_data2"]),
        min(assignments["train_loader2"]),
        min(assignments["val_loader2"]),
    ]
    assert order == sorted(order)

    source = _compact_function_source(RUNNER_PATH, "build_data_loaders")
    assert source.count("shuffle=True") == 2
    assert source.count("shuffle=False") == 3
    assert source.count("drop_last=True") == 4
    assert source.count("drop_last=False") == 1
    assert "val_loader2.complete_au_loader=" in source
    assert "batch_size=au_batch_size" in source
    assert '9ifau_name=="bp4d"else8' in source
