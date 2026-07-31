import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# AU and subject orders are fixed by the released three-fold protocol.
DISFA_AUS = (1, 2, 4, 6, 9, 12, 25, 26)
DISFA_SUBJECTS = (
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
)


def build_disfa_transform(is_train, args):
    """Build the DISFA frame transform."""
    # DISFA keeps its original ImageNet-normalized per-frame recipe.
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.3,
                    scale=(0.02, 0.1),
                    ratio=(0.5, 2.0),
                    value="random",
                ),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform


def build_disfa_dataset(json_path, is_train, args):
    """Build the DISFA sequence dataset."""
    transform = build_disfa_transform(is_train, args)
    dataset = DisfaSequenceDataset(
        args.root_path,
        json_path,
        transform=transform,
    )
    print(dataset)
    return dataset


class DisfaSequenceDataset(Dataset):
    """Pack DISFA JSONL frame records into verified 16-frame sequences."""

    def __init__(
        self,
        root_path,
        json_file,
        transform=None,
        clip_len=16,
        temporal_step=1,
        stride=16,
        skip_missing=True,
        verify_images=False,
        verbose=True,
    ):
        super().__init__()

        if clip_len <= 0:
            raise ValueError(
                f"clip_len must be greater than zero, got {clip_len}"
            )

        if temporal_step <= 0:
            raise ValueError(
                "temporal_step must be greater than zero, "
                f"got {temporal_step}"
            )

        if stride <= 0:
            raise ValueError(
                f"stride must be greater than zero, got {stride}"
            )

        self.root_path = os.path.abspath(root_path)
        self.json_file = json_file
        self.transform = transform

        self.clip_len = clip_len
        self.temporal_step = temporal_step
        self.stride = stride
        self.required_span = (
            (self.clip_len - 1) * self.temporal_step + 1
        )

        self.skip_missing = skip_missing
        self.verify_images = verify_images
        self.verbose = verbose

        self.au_ids = list(DISFA_AUS)
        self.subject_ids = list(DISFA_SUBJECTS)

        self.au_to_index = {
            au: index for index, au in enumerate(self.au_ids)
        }
        self.subject_to_index = {
            subject_id: index
            for index, subject_id in enumerate(self.subject_ids)
        }

        self.data = self._load_data(json_file)

        self.missing_files = []
        self.corrupted_files = []
        self.invalid_records = []
        self.unknown_subjects = []

        self.samples = self._build_samples()

        if self.verbose:
            self._print_summary()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid DISFA temporal samples were constructed.\n"
                f"clip_len={self.clip_len}, "
                f"temporal_step={self.temporal_step}, "
                f"required continuous span={self.required_span} frames.\n"
                "Check the image root, JSONL path, frame naming, and data "
                "completeness."
            )

    def _load_data(self, json_file):
        # The annotations are JSON Lines: one frame object per line.
        data = []

        with open(json_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Failed to parse JSONL line {line_number}: {error}"
                    ) from error

                if not isinstance(item, dict):
                    raise ValueError(
                        f"JSONL line {line_number} is not an object: "
                        f"{type(item)}"
                    )

                data.append(item)

        return data

    def _extract_subject_id(self, img_path):
        match = re.search(
            r"(SN\d{3})",
            str(img_path),
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        return match.group(1).upper()

    def _extract_frame_index(self, img_path):
        filename = os.path.basename(str(img_path))
        stem, _ = os.path.splitext(filename)
        numbers = re.findall(r"\d+", stem)

        if not numbers:
            return None

        return int(numbers[-1])

    def _resolve_image_path(self, img_path):
        img_path = str(img_path).strip()

        if os.path.isabs(img_path) and os.path.isfile(img_path):
            return os.path.abspath(img_path)

        relative_path = img_path.lstrip("/\\")
        relative_path = relative_path.replace(
            "/",
            os.sep,
        ).replace(
            "\\",
            os.sep,
        )

        return os.path.abspath(
            os.path.join(self.root_path, relative_path)
        )

    def _get_sequence_name(self, img_path):
        normalized_path = str(img_path).replace("\\", "/")
        return os.path.dirname(normalized_path)

    def _is_valid_image(self, image_path):
        if not os.path.isfile(image_path):
            return False, "missing"

        if self.verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                return False, "corrupted"

        return True, None

    def _split_continuous_segments(self, frame_records):
        if not frame_records:
            return []

        frame_records = sorted(
            frame_records,
            key=lambda record: record["frame_index"],
        )

        continuous_segments = []
        current_segment = [frame_records[0]]

        for current_record in frame_records[1:]:
            previous_record = current_segment[-1]
            previous_index = previous_record["frame_index"]
            current_index = current_record["frame_index"]

            if current_index == previous_index + 1:
                current_segment.append(current_record)
            else:
                continuous_segments.append(current_segment)
                current_segment = [current_record]

        continuous_segments.append(current_segment)
        return continuous_segments

    def _build_samples(self):
        # Group by subject and sequence before checking temporal continuity.
        grouped_frames = defaultdict(list)
        valid_frame_count = 0

        for record_index, item in enumerate(self.data):
            if "img_path" not in item:
                self.invalid_records.append(
                    {
                        "record_index": record_index,
                        "reason": "missing img_path field",
                        "item": item,
                    }
                )
                continue

            if "AUs" not in item:
                self.invalid_records.append(
                    {
                        "record_index": record_index,
                        "reason": "missing AUs field",
                        "item": item,
                    }
                )
                continue

            img_path = item["img_path"]
            subject_id = self._extract_subject_id(img_path)
            frame_index = self._extract_frame_index(img_path)
            sequence_name = self._get_sequence_name(img_path)

            if subject_id is None:
                self.invalid_records.append(
                    {
                        "record_index": record_index,
                        "reason": "subject ID cannot be parsed from path",
                        "img_path": img_path,
                    }
                )
                continue

            if subject_id not in self.subject_to_index:
                self.unknown_subjects.append(
                    {
                        "record_index": record_index,
                        "subject_id": subject_id,
                        "img_path": img_path,
                    }
                )
                continue

            if frame_index is None:
                self.invalid_records.append(
                    {
                        "record_index": record_index,
                        "reason": "frame index cannot be parsed from filename",
                        "img_path": img_path,
                    }
                )
                continue

            if not sequence_name:
                self.invalid_records.append(
                    {
                        "record_index": record_index,
                        "reason": "sequence directory cannot be parsed",
                        "img_path": img_path,
                    }
                )
                continue

            image_path = self._resolve_image_path(img_path)
            is_valid, invalid_type = self._is_valid_image(image_path)

            if not is_valid:
                invalid_info = {
                    "record_index": record_index,
                    "img_path": img_path,
                    "resolved_path": image_path,
                }

                if invalid_type == "missing":
                    self.missing_files.append(invalid_info)
                else:
                    self.corrupted_files.append(invalid_info)

                continue

            frame_record = {
                "item": item,
                "subject_id": subject_id,
                "sequence_name": sequence_name,
                "frame_index": frame_index,
                "image_path": image_path,
            }
            group_key = (
                subject_id,
                sequence_name,
            )
            grouped_frames[group_key].append(frame_record)
            valid_frame_count += 1

        if not self.skip_missing:
            errors = []

            if self.missing_files:
                errors.append(
                    f"Missing images: {len(self.missing_files)}"
                )

            if self.corrupted_files:
                errors.append(
                    f"Corrupted images: {len(self.corrupted_files)}"
                )

            if errors:
                preview = []

                for info in self.missing_files[:10]:
                    preview.append(
                        f"Missing: {info['resolved_path']}"
                    )

                for info in self.corrupted_files[:10]:
                    preview.append(
                        f"Corrupted: {info['resolved_path']}"
                    )

                raise FileNotFoundError(
                    "DISFA data validation failed:\n"
                    + "\n".join(errors)
                    + "\n"
                    + "\n".join(preview)
                )

        samples = []
        continuous_segment_count = 0
        short_segment_count = 0

        for group_key, frame_records in grouped_frames.items():
            continuous_segments = self._split_continuous_segments(
                frame_records
            )
            continuous_segment_count += len(continuous_segments)

            for segment in continuous_segments:
                if len(segment) < self.required_span:
                    short_segment_count += 1
                    continue

                max_start = len(segment) - self.required_span

                # Pack clips with the historical 16-frame start stride.
                for start in range(
                    0,
                    max_start + 1,
                    self.stride,
                ):
                    clip = [
                        segment[
                            start + i * self.temporal_step
                        ]
                        for i in range(self.clip_len)
                    ]

                    subject_ids = {
                        frame["subject_id"] for frame in clip
                    }
                    sequence_names = {
                        frame["sequence_name"] for frame in clip
                    }
                    frame_indices = [
                        frame["frame_index"] for frame in clip
                    ]
                    expected_indices = [
                        frame_indices[0]
                        + i * self.temporal_step
                        for i in range(self.clip_len)
                    ]

                    if len(subject_ids) != 1:
                        continue

                    if len(sequence_names) != 1:
                        continue

                    if frame_indices != expected_indices:
                        continue

                    samples.append(clip)

        self.valid_frame_count = valid_frame_count
        self.continuous_segment_count = continuous_segment_count
        self.short_segment_count = short_segment_count

        return samples

    def _print_summary(self):
        print("=" * 70)
        print("DISFA dataset validation summary")
        print(f"JSON records:          {len(self.data)}")
        print(f"Valid image frames:    {self.valid_frame_count}")
        print(f"Missing images:        {len(self.missing_files)}")
        print(f"Corrupted images:      {len(self.corrupted_files)}")
        print(f"Invalid records:       {len(self.invalid_records)}")
        print(f"Unknown subjects:      {len(self.unknown_subjects)}")
        print(
            f"Continuous segments:   "
            f"{self.continuous_segment_count}"
        )
        print(
            f"Short segments:        "
            f"{self.short_segment_count}"
        )
        print(f"Clip length:           {self.clip_len}")
        print(f"Temporal step:         {self.temporal_step}")
        print(f"Raw-frame span:        {self.required_span}")
        print(f"Clip start stride:     {self.stride}")
        print(f"Valid clips:           {len(self.samples)}")

        if self.missing_files:
            print("-" * 70)
            print("Missing image examples:")

            for info in self.missing_files[:20]:
                print(f"  {info['resolved_path']}")

            remaining = len(self.missing_files) - 20

            if remaining > 0:
                print(f"  ... and {remaining} more")

        if self.corrupted_files:
            print("-" * 70)
            print("Corrupted image examples:")

            for info in self.corrupted_files[:20]:
                print(f"  {info['resolved_path']}")

            remaining = len(self.corrupted_files) - 20

            if remaining > 0:
                print(f"  ... and {remaining} more")

        print("=" * 70)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # DISFA intentionally samples transforms independently per frame.
        clip = self.samples[idx]

        images = []
        au_label_sequence = []
        identity_label_sequence = []

        for frame_record in clip:
            frame_info = frame_record["item"]
            image_path = frame_record["image_path"]
            subject_id = frame_record["subject_id"]

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    "Image was removed or moved after dataset "
                    f"initialization:\n{image_path}"
                )

            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

            images.append(image)

            au_labels_frame = torch.zeros(
                len(self.au_ids),
                dtype=torch.float32,
            )
            frame_aus = frame_info.get("AUs", [])

            # The annotation generator uses 999 as an inactive placeholder.
            for au in frame_aus:
                try:
                    au = int(au)
                except (TypeError, ValueError):
                    continue

                if au == 999:
                    continue

                if au in self.au_to_index:
                    au_index = self.au_to_index[au]
                    au_labels_frame[au_index] = 1.0

            au_label_sequence.append(au_labels_frame)

            identity_labels_frame = torch.zeros(
                len(self.subject_ids),
                dtype=torch.float32,
            )
            subject_index = self.subject_to_index[subject_id]
            identity_labels_frame[subject_index] = 1.0
            identity_label_sequence.append(identity_labels_frame)

        stacked_images = torch.stack(
            images,
            dim=0,
        )
        au_label_sequence = torch.stack(
            au_label_sequence,
            dim=0,
        )
        identity_label_sequence = torch.stack(
            identity_label_sequence,
            dim=0,
        )

        return stacked_images, (
            au_label_sequence,
            identity_label_sequence,
        )


# Keep the public loader surface explicit for downstream imports.
__all__ = [
    "DISFA_AUS",
    "DISFA_SUBJECTS",
    "DisfaSequenceDataset",
    "build_disfa_dataset",
    "build_disfa_transform",
]
