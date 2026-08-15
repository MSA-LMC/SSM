# Guard DISFA loading, smoothing, training, and metric parity.
import ast
import json
import random
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ssm.config import load_config  # noqa: E402
from ssm.data.disfa import (  # noqa: E402
    DISFA_AUS,
    DISFA_SUBJECTS,
    DisfaSequenceDataset,
    build_disfa_transform,
)
from ssm.semantics import (  # noqa: E402
    DISFA_AU_DESCRIPTIONS,
    DISFA_EMOTION_DESCRIPTIONS,
)
import ssm.training.legacy as legacy  # noqa: E402


DATA_SOURCE = REPOSITORY_ROOT / "ssm" / "data" / "disfa.py"
MODEL_SOURCE = REPOSITORY_ROOT / "ssm" / "models" / "disfa.py"
TRAINING_SOURCE = REPOSITORY_ROOT / "ssm" / "training" / "legacy.py"
METRICS_SOURCE = REPOSITORY_ROOT / "ssm" / "evaluation" / "metrics.py"
ADAPTIVE_VOTE_SOURCE = REPOSITORY_ROOT / "ssm" / "evaluation" / "adaptive_vote.py"
RUNNER_SOURCE = REPOSITORY_ROOT / "ssm" / "runner.py"
TRAIN_ENTRY_SOURCE = REPOSITORY_ROOT / "train.py"


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(path, name, class_name=None):
    tree = _parse(path)
    body = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    return next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _compact_function(path, name, class_name=None):
    node = _find_function(path, name, class_name)
    return "".join(ast.unparse(node).split())


def _compile_functions(path, names, namespace):
    tree = _parse(path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _binary_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float(2 * true_positive / denominator)


def _binary_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both classes are required.")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size
    )


class _SequentialTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return torch.full(
            (3, 2, 2),
            float(self.calls),
            dtype=torch.float32,
        )


class _SizedLoader(list):
    def __init__(self, batches, dataset_size):
        super().__init__(batches)
        self.dataset = range(dataset_size)


class DisfaDataCompatibilityTests(unittest.TestCase):
    def test_constants_and_prompt_order_match_the_release_protocol(self):
        self.assertEqual(DISFA_AUS, (1, 2, 4, 6, 9, 12, 25, 26))
        self.assertEqual(
            DISFA_SUBJECTS,
            (
                "SN001",
                "SN002",
                "SN009",
                "SN010",
                "SN016",
                "SN026",
                "SN027",
                "SN030",
                "SN032",
                "SN006",
                "SN011",
                "SN012",
                "SN013",
                "SN018",
                "SN021",
                "SN024",
                "SN028",
                "SN031",
                "SN003",
                "SN004",
                "SN005",
                "SN007",
                "SN008",
                "SN017",
                "SN023",
                "SN025",
                "SN029",
            ),
        )
        self.assertEqual(
            DISFA_AU_DESCRIPTIONS,
            [
                "inner brow raiser",
                "outer brow raiser",
                "brow lowerer",
                "cheek raiser",
                "nose wrinkler",
                "lip corner puller",
                "lips part",
                "jaw drop",
            ],
        )
        self.assertEqual(
            DISFA_EMOTION_DESCRIPTIONS,
            [
                "happiness, cheek raiser, lip corner puller",
                "sadness, inner brow raiser, brow lowerer",
                (
                    "neutral, relaxed facial muscles, straight mouth, "
                    "smooth forehead, unremarkable eyebrows"
                ),
                "anger, brow lowerer",
                ("surprise, inner brow raiser, outer brow raiser, jaw drop"),
                "disgust, nose wrinkler",
                ("fear, inner brow raiser, outer brow raiser, brow lowerer, jaw drop"),
            ],
        )

    def test_transform_recipe_and_rng_match_the_legacy_recipe(self):
        args = SimpleNamespace()
        actual = build_disfa_transform(True, args)
        self.assertTrue(actual.is_train)
        self.assertEqual(actual.image_size, 224)
        self.assertEqual(actual.resize_size, 256)
        self.assertEqual(tuple(actual.crop_scale), (0.8, 1.0))
        self.assertEqual(tuple(actual.crop_ratio), (0.9, 1.1))
        self.assertEqual(actual.horizontal_flip_probability, 0.5)
        self.assertEqual(
            tuple(actual.normalize.mean),
            (0.48145466, 0.4578275, 0.40821073),
        )
        self.assertEqual(
            tuple(actual.normalize.std),
            (0.26862954, 0.26130258, 0.27577711),
        )

        y_grid, x_grid = np.mgrid[:280, :300]
        image_array = np.stack(
            [
                x_grid % 256,
                y_grid % 256,
                (x_grid + y_grid) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        image = Image.fromarray(image_array)

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        clip_tensor = actual.transform_clip([image, image.copy()])
        self.assertEqual(tuple(clip_tensor.shape), (2, 3, 224, 224))
        self.assertTrue(torch.equal(clip_tensor[0], clip_tensor[1]))

        evaluation = build_disfa_transform(False, args)
        self.assertFalse(evaluation.is_train)
        evaluation_tensor = evaluation.transform_clip([image, image.copy()])
        self.assertEqual(
            tuple(evaluation_tensor.shape),
            (2, 3, 224, 224),
        )
        self.assertTrue(torch.equal(evaluation_tensor[0], evaluation_tensor[1]))

    def test_jsonl_sequence_packing_labels_and_per_frame_transform(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            image_root = temporary_root / "images"
            sequence_root = image_root / "RightVideoSN001"
            sequence_root.mkdir(parents=True)
            annotation_path = temporary_root / "disfa.jsonl"

            with annotation_path.open("w", encoding="utf-8") as annotation:
                for frame_index in range(1, 34):
                    image_path = sequence_root / f"frame_{frame_index}.jpg"
                    Image.new(
                        "RGB",
                        (8, 8),
                        color=(frame_index, 0, 0),
                    ).save(image_path)
                    if frame_index == 1:
                        action_units = [1, "12", 999, "invalid"]
                    elif frame_index == 2:
                        action_units = [2, 4]
                    else:
                        action_units = []
                    annotation.write(
                        json.dumps(
                            {
                                "img_path": (
                                    f"RightVideoSN001/frame_{frame_index}.jpg"
                                ),
                                "AUs": action_units,
                            }
                        )
                        + "\n"
                    )

            transform = _SequentialTransform()
            dataset = DisfaSequenceDataset(
                root_path=str(image_root),
                json_file=str(annotation_path),
                transform=transform,
                verbose=False,
            )

            self.assertEqual(dataset.clip_len, 16)
            self.assertEqual(dataset.temporal_step, 1)
            self.assertEqual(dataset.stride, 16)
            self.assertEqual(dataset.required_span, 16)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                [record["frame_index"] for record in dataset.samples[0]],
                list(range(1, 17)),
            )
            self.assertEqual(
                [record["frame_index"] for record in dataset.samples[1]],
                list(range(17, 33)),
            )

            images, (au_labels, identity_labels) = dataset[0]
            self.assertEqual(images.shape, (16, 3, 2, 2))
            for frame_index in range(16):
                self.assertTrue(
                    torch.equal(
                        images[frame_index],
                        torch.full(
                            (3, 2, 2),
                            float(frame_index + 1),
                        ),
                    )
                )
            self.assertEqual(transform.calls, 16)
            self.assertEqual(au_labels.shape, (16, 8))
            self.assertTrue(
                torch.equal(
                    au_labels[0],
                    torch.tensor(
                        [1, 0, 0, 0, 0, 1, 0, 0],
                        dtype=torch.float32,
                    ),
                )
            )
            self.assertTrue(
                torch.equal(
                    au_labels[1],
                    torch.tensor(
                        [0, 1, 1, 0, 0, 0, 0, 0],
                        dtype=torch.float32,
                    ),
                )
            )
            self.assertEqual(identity_labels.shape, (16, 27))
            self.assertTrue(torch.all(identity_labels[:, 0] == 1))
            self.assertEqual(float(identity_labels.sum()), 16.0)

    def test_complete_evaluation_skips_missing_frames_without_crossing_gaps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            image_root = temporary_root / "images"
            sequence_root = image_root / "RightVideoSN001"
            sequence_root.mkdir(parents=True)
            annotation_path = temporary_root / "disfa.jsonl"

            with annotation_path.open("w", encoding="utf-8") as annotation:
                for frame_index in range(1, 8):
                    if frame_index != 4:
                        Image.new("RGB", (8, 8)).save(
                            sequence_root / f"frame_{frame_index}.jpg"
                        )
                    annotation.write(
                        json.dumps(
                            {
                                "img_path": (
                                    f"RightVideoSN001/frame_{frame_index}.jpg"
                                ),
                                "AUs": [],
                            }
                        )
                        + "\n"
                    )

            dataset = DisfaSequenceDataset(
                root_path=str(image_root),
                json_file=str(annotation_path),
                transform=_SequentialTransform(),
                clip_len=4,
                stride=4,
                skip_missing=True,
                verbose=False,
                complete_evaluation=True,
            )

            self.assertEqual(len(dataset.missing_files), 1)
            self.assertEqual(dataset.valid_frame_count, 6)
            self.assertEqual(dataset.track_frame_count, 6)
            self.assertEqual(dataset.continuous_segment_count, 2)
            self.assertEqual(dataset.segment_count_with_targets, 2)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                [
                    [record["frame_index"] for record in sample]
                    for sample in dataset.samples
                ],
                [[1, 2, 3, 3], [5, 6, 7, 7]],
            )
            self.assertEqual(
                [metadata["target_mask"] for metadata in dataset.eval_metadata],
                [[True, True, True, False]] * 2,
            )
            self.assertNotEqual(
                dataset.eval_metadata[0]["segment_id"],
                dataset.eval_metadata[1]["segment_id"],
            )

            self.assertEqual(dataset.pad_complete_evaluation(4), 2)
            self.assertEqual(len(dataset), 4)

            covered_frames = []
            for sample_index in range(len(dataset)):
                metadata = dataset.get_eval_metadata(sample_index)
                covered_frames.extend(
                    frame_index
                    for frame_index, is_target in zip(
                        metadata["frame_indices"],
                        metadata["target_mask"],
                    )
                    if is_target
                )

            self.assertEqual(covered_frames, [1, 2, 3, 5, 6, 7])
            self.assertEqual(len(covered_frames), len(set(covered_frames)))

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing images: 1",
            ):
                DisfaSequenceDataset(
                    root_path=str(image_root),
                    json_file=str(annotation_path),
                    transform=_SequentialTransform(),
                    clip_len=4,
                    stride=4,
                    skip_missing=False,
                    verbose=False,
                    complete_evaluation=True,
                )


class DisfaTrainingCompatibilityTests(unittest.TestCase):
    def test_clip_mixup_uses_one_partner_for_all_frames(self):
        inputs = torch.arange(48, dtype=torch.float32).reshape(4, 2, 3, 1, 2)
        targets = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)

        np.random.seed(73)
        torch.manual_seed(41)
        mixed, targets_a, targets_b, lam = legacy.mixup_clips(
            inputs,
            targets,
            alpha=0.4,
        )

        np.random.seed(73)
        torch.manual_seed(41)
        expected_lam = np.random.beta(0.4, 0.4)
        frame_permutation = torch.randperm(inputs.size(0) * inputs.size(1))
        expected_order = []
        for flat_index in frame_permutation.tolist():
            clip_index = flat_index // inputs.size(1)
            if clip_index not in expected_order:
                expected_order.append(clip_index)
        expected_index = torch.tensor(expected_order)
        expected_mixed = (
            expected_lam * inputs + (1 - expected_lam) * inputs[expected_index, :]
        )

        self.assertEqual(lam, expected_lam)
        self.assertTrue(torch.equal(mixed, expected_mixed))
        self.assertTrue(torch.equal(targets_a, targets))
        self.assertTrue(torch.equal(targets_b, targets[expected_index]))

    def test_identity_labels_are_moved_and_kept_alive(self):
        action_units = torch.zeros(2, 16, 8)
        identities = torch.zeros(2, 16, 27)
        device = torch.device("cpu")

        with mock.patch.object(
            legacy,
            "move_to_runtime",
            side_effect=lambda tensor, runtime: tensor,
        ) as move:
            result_action_units, result_identities = legacy._unpack_au_targets(
                (action_units, identities),
                "disfa",
                device,
            )

        self.assertIs(result_action_units, action_units)
        self.assertIs(result_identities, identities)
        self.assertEqual(
            move.call_args_list,
            [
                mock.call(action_units, device),
                mock.call(identities, device),
            ],
        )

    def test_ema_update_apply_and_restore_match_the_legacy_equations(self):
        model = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0]]))
            model.bias.copy_(torch.tensor([0.5]))

        ema = legacy.ModelEMA(model, decay=0.99)
        initial = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(3.0)
        updated = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        ema.update()
        expected_shadow = {
            name: 0.99 * initial[name] + 0.01 * updated[name] for name in initial
        }
        for name in expected_shadow:
            self.assertTrue(torch.equal(ema.shadow[name], expected_shadow[name]))

        ema.apply_shadow()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, expected_shadow[name]))
        ema.restore()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, updated[name]))

    def test_joint_step_ast_preserves_clip_mixup(self):
        source = _compact_function(
            TRAINING_SOURCE,
            "_joint_training_step",
        )
        required_fragments = [
            "mixup_clips(au_inputs,au_labels1,alpha=mixup_alpha)",
            "targets_a_flat=targets_a.reshape(-1,au_count)",
            "targets_b_flat=targets_b.reshape(-1,au_count)",
            (
                "loss2=lam*criterion2(au_output,targets_a_flat)"
                "+(1-lam)*criterion2(au_output,targets_b_flat)"
            ),
            (
                "loss4=lam*criterion2(au_output_pro,targets_a_flat)"
                "+(1-lam)*criterion2(au_output_pro,targets_b_flat)"
            ),
            "momentum=0.95",
            "eps=1e-08",
            "p=epoch/30",
            "a=2/(1+np.exp(-10*p))-1",
            (
                "total_loss=(loss1_scaled+2.0*loss2_scaled"
                "+a*(loss3_scaled+loss4_scaled))/3.0"
            ),
            "torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)",
            "optimizer.step()",
            "ema.update()",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, source)

    def test_runner_order_and_normalized_disfa_defaults(self):
        config = load_config(REPOSITORY_ROOT / "configs" / "disfa_dfew.json")
        namespace = {"SimpleNamespace": SimpleNamespace}
        functions = _compile_functions(
            RUNNER_SOURCE,
            {"build_runtime_args"},
            namespace,
        )
        args = functions["build_runtime_args"](config)
        self.assertEqual(args.dataset, "DFEW")
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch_size, 12)
        self.assertEqual(args.print_freq, 100)
        self.assertEqual(args.milestones, [15, 25])
        self.assertEqual(args.contexts_number, 8)
        self.assertEqual(args.class_token_position, "end")
        self.assertEqual(args.class_specific_contexts, "True")
        self.assertEqual(args.temporal_layers, 1)
        self.assertEqual(args.smooth_K, 2)
        self.assertEqual(args.input_size, 224)
        self.assertEqual(config["runtime"]["gpu_ids"], [0, 1, 2])

        source = _compact_function(RUNNER_SOURCE, "train_fold_pair")
        ordered_fragments = [
            "set_reproducible_seed(args.seed)",
            "model=build_joint_model(config,args,device)",
            "ema=ModelEMA(core_model,decay=0.99)",
            "optimizer=build_legacy_optimizer(core_model)",
            "scheduler.step()",
            "ema.apply_shadow()",
            "val_acc,val_los=validate_joint(",
            "evaluate_action_units(",
            "ema.restore()",
            "load_checkpoint_state(model,paths['best_emotion'],device)",
            "evaluate_emotions(",
        ]
        positions = [source.index(fragment) for fragment in ordered_fragments]
        self.assertEqual(positions, sorted(positions))

        validation_source = _compact_function(
            TRAINING_SOURCE,
            "validate_joint",
        )
        self.assertIn(
            "enumerate(zip(val_loader1,val_loader2))",
            validation_source,
        )

        entry_source = _compact_function(TRAIN_ENTRY_SOURCE, "main")
        self.assertIn(
            "all_fold_pairs(config['experiment']['emotion_dataset'])",
            entry_source,
        )
        all_folds_source = _compact_function(
            TRAIN_ENTRY_SOURCE,
            "all_fold_pairs",
        )
        self.assertIn(
            "foremotion_foldin[5,4,3,2,1]forau_foldin[1,2,3]",
            all_folds_source,
        )
        self.assertLess(
            entry_source.index(
                "os.environ['CUDA_VISIBLE_DEVICES']=str(visible_devices)"
            ),
            entry_source.index("fromssm.runnerimporttrain_all"),
        )


class DisfaSmoothingAndEvaluationCompatibilityTests(unittest.TestCase):
    def test_disfa_smoothing_ast_preserves_pairwise_depthwise_filter(self):
        constructor = _compact_function(
            MODEL_SOURCE,
            "__init__",
            class_name="DisfaSSM",
        )
        forward = _compact_function(
            MODEL_SOURCE,
            "forward",
            class_name="DisfaSSM",
        )
        constructor_fragments = [
            "K=getattr(args,'smooth_K',2)",
            "window=2*K+1",
            (
                "self.au_smooth=nn.Conv1d(in_channels=8,"
                "out_channels=8,kernel_size=window,padding=K,"
                "groups=8,bias=False)"
            ),
            "nn.init.constant_(self.au_smooth.weight,1.0/window)",
        ]
        for fragment in constructor_fragments:
            self.assertIn(fragment, constructor)

        forward_fragments = [
            "combined_au=combined_au.reshape(B2,16,512)",
            "au_logits=au_logits_flat.view(B2*16//2,2,8)",
            "smooth_in=au_logits.permute(0,2,1)",
            "smooth_out=self.au_smooth(smooth_in)",
            "au_logits_sm=smooth_out.permute(0,2,1)",
            "au_logits_sm=au_logits_sm.reshape(-1,8)",
            "0.1*au_logits_flat_2",
        ]
        for fragment in forward_fragments:
            self.assertIn(fragment, forward)

    def test_majority_voting_keeps_edge_padding_behavior(self):
        namespace = {"np": np}
        functions = _compile_functions(
            METRICS_SOURCE,
            {"majority_voting"},
            namespace,
        )
        predictions = np.array(
            [
                [0, 1],
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 1],
            ],
            dtype=int,
        )
        expected = np.array(
            [
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
            ],
            dtype=int,
        )
        actual = functions["majority_voting"](
            predictions,
            K=1,
            p=0.5,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_au_evaluation_uses_complete_segment_safe_joint_search(self):
        source = _compact_function(
            METRICS_SOURCE,
            "evaluate_action_units",
        )
        required_fragments = [
            "complete_loader=getattr(val_loader2,'complete_au_loader',None)",
            "forau_inputs,au_labelsincomplete_loader",
            "output=model(dfer_inputs,au_inputs)[2]",
            "metadata=dataset.get_eval_metadata(sample_cursor+row)",
            "search_segment_safe_vote(",
            "p_values=_complete_p_candidates(K_values)",
            "expected_au_count=int(au_count)",
            "roc_auc_score(all_targets[:,au_index],",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, source)

    def test_au_evaluation_reconstructs_a_complete_disfa_segment(self):
        adaptive_namespace = runpy.run_path(str(ADAPTIVE_VOTE_SOURCE))
        namespace = {
            "OrderedDict": __import__("collections").OrderedDict,
            "F": torch.nn.functional,
            "SequentialSampler": torch.utils.data.SequentialSampler,
            "move_to_runtime": lambda tensor, device: tensor.to(device),
            "np": np,
            "roc_auc_score": _binary_auc,
            "search_segment_safe_vote": adaptive_namespace["search_segment_safe_vote"],
            "torch": torch,
        }
        functions = _compile_functions(
            METRICS_SOURCE,
            {
                "_inclusive_float_grid",
                "_complete_p_candidates",
                "evaluate_action_units",
            },
            namespace,
        )
        evaluate_action_units = functions["evaluate_action_units"]

        class CompleteDataset(torch.utils.data.Dataset):
            track_frame_count = 2
            segment_count_with_targets = 1

            def __len__(self):
                return 1

            def __getitem__(self, index):
                del index
                targets = torch.tensor([[0.0] * 8, [1.0] * 8])
                identities = torch.zeros(2, 27)
                identities[:, 0] = 1.0
                return torch.zeros(2, 3, 1, 1), (
                    targets,
                    identities,
                )

            def get_eval_metadata(self, index):
                self.assert_index(index)
                return {
                    "segment_id": "SN001:RightVideoSN001:1-2:1",
                    "target_mask": [True, True],
                    "frame_indices": [1, 2],
                }

            @staticmethod
            def assert_index(index):
                if int(index) != 0:
                    raise AssertionError(index)

        complete_loader = torch.utils.data.DataLoader(
            CompleteDataset(),
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        au_loader = SimpleNamespace(complete_au_loader=complete_loader)
        dfer_loader = [(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))]

        class Model:
            def __init__(self):
                self.call_count = 0

            def eval(self):
                return self

            def __call__(self, dfer_inputs, au_inputs):
                del dfer_inputs, au_inputs
                self.call_count += 1
                logits = torch.tensor([[-4.0] * 8, [4.0] * 8])
                return None, None, logits, None, None, None

        model = Model()
        result, mean_f1, mean_auc = evaluate_action_units(
            dfer_loader,
            au_loader,
            model,
            torch.device("cpu"),
            "disfa",
            8,
            evaluation_config={
                "au_threshold": {
                    "search_start": 0.1,
                    "search_stop": 0.9,
                    "search_step": 0.1,
                },
                "adaptive_vote": {
                    "half_window_min": 0,
                    "half_window_max": 0,
                    "tie_reference": {
                        "half_window": 0,
                        "positive_ratio": 1.0,
                        "threshold": 0.5,
                    },
                },
            },
        )

        self.assertEqual(model.call_count, 1)
        self.assertEqual(result["evaluated_frames"], 2)
        self.assertEqual(result["selection_scope"], "current_evaluation_labels")
        self.assertAlmostEqual(mean_f1, 1.0)
        self.assertAlmostEqual(mean_auc, 1.0)

    def test_emotion_evaluation_keeps_zip_truncation_and_war_denominator(self):
        def confusion_matrix(y_true, y_pred):
            y_true = np.asarray(y_true).reshape(-1).astype(int)
            y_pred = np.asarray(y_pred).reshape(-1).astype(int)
            size = int(max(y_true.max(), y_pred.max())) + 1
            matrix = np.zeros((size, size), dtype=int)
            for target, prediction in zip(y_true, y_pred):
                matrix[target, prediction] += 1
            return matrix

        namespace = {
            "np": np,
            "torch": torch,
            "confusion_matrix": confusion_matrix,
            "move_to_runtime": lambda tensor, device: tensor,
        }
        functions = _compile_functions(
            METRICS_SOURCE,
            {"evaluate_emotions"},
            namespace,
        )

        class Model:
            def __init__(self):
                self.call_count = 0

            def eval(self):
                return self

            def __call__(self, dfer_inputs, au_inputs):
                self.call_count += 1
                logits = torch.tensor(
                    [
                        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
                return (logits,)

        dfer_loader = _SizedLoader(
            [
                (
                    torch.zeros(2, 1),
                    torch.tensor([0, 1], dtype=torch.long),
                ),
                (
                    torch.zeros(2, 1),
                    torch.tensor([0, 1], dtype=torch.long),
                ),
            ],
            dataset_size=5,
        )
        au_loader = _SizedLoader(
            [(torch.zeros(2, 1), torch.zeros(2, 1))],
            dataset_size=2,
        )
        model = Model()
        uar, war = functions["evaluate_emotions"](
            dfer_loader,
            au_loader,
            model,
            torch.device("cpu"),
            [str(index) for index in range(7)],
            "DFEW",
            1,
            confusion_matrix_path=None,
            log_txt_path=None,
        )

        self.assertEqual(model.call_count, 1)
        self.assertAlmostEqual(float(uar), 100.0)
        self.assertAlmostEqual(float(war), 40.0)


if __name__ == "__main__":
    unittest.main()
