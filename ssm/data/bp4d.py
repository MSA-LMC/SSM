import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def pil_loader(path: str) -> Image.Image:
    """Load one image as RGB."""
    with open(path, "rb") as file:
        image = Image.open(file)
        return image.convert("RGB")


class Bp4dSequenceDataset(Dataset):
    """Pack consecutive BP4D frames into non-overlapping 16-frame clips."""

    # Label order and subject identities follow the released BP4D protocol.
    DEFAULT_AU_IDS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

    DEFAULT_SUBJECT_IDS = [
        "F001", "F002", "F003", "F004", "F005", "F006", "F007",
        "F008", "F009", "F010", "F011", "F012", "F013", "F014",
        "F015", "F016", "F017", "F018", "F019", "F020", "F021",
        "F022", "F023",
        "M001", "M002", "M003", "M004", "M005", "M006", "M007",
        "M008", "M009", "M010", "M011", "M012", "M013", "M014",
        "M015", "M016", "M017", "M018",
    ]

    def __init__(
        self,
        label_file: str,
        root: Optional[str] = None,
        is_train: bool = True,
        transform=None,
        loader=pil_loader,
        aus_list: Optional[Sequence[int]] = None,
        clip_len: int = 16,
        stride: int = 16,
        skip_invalid: bool = True,
        verify_images: bool = False,
        sync_transform: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        if clip_len <= 0:
            raise ValueError(f"clip_len must be positive, got {clip_len}.")

        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")

        self.label_file = os.path.abspath(label_file)
        self.root = os.path.abspath(root) if root is not None else None
        self.is_train = is_train
        self.loader = loader
        self.clip_len = clip_len
        self.stride = stride
        self.skip_invalid = skip_invalid
        self.verify_images = verify_images
        self.sync_transform = sync_transform
        self.verbose = verbose

        self.au_ids = (
            list(aus_list)
            if aus_list is not None
            else list(self.DEFAULT_AU_IDS)
        )
        self.subject_ids = list(self.DEFAULT_SUBJECT_IDS)

        self.num_aus = len(self.au_ids)
        self.au_to_index = {
            au: index for index, au in enumerate(self.au_ids)
        }
        self.subject_to_index = {
            subject_id: index
            for index, subject_id in enumerate(self.subject_ids)
        }

        # Default transforms match the original BP4D train/test branches.
        if transform is None:
            self.transform = self._build_default_transform(is_train)
        else:
            self.transform = transform

        self.samples: List[List[Dict]] = []

        self.total_records = 0
        self.valid_frame_count = 0
        self.missing_files: List[Dict] = []
        self.corrupted_files: List[Dict] = []
        self.invalid_records: List[Dict] = []
        self.duplicate_frames: List[Dict] = []
        self.continuous_segment_count = 0
        self.short_segment_count = 0
        self.sequence_count = 0

        self._build_samples()

        if self.verbose:
            self._print_summary()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid BP4D 16-frame clips were constructed.\n"
                "Check the label file, image paths, sequence-folder format, "
                "frame numbering, and image completeness."
            )

    @staticmethod
    def _build_default_transform(is_train: bool):
        if is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.481, 0.457, 0.408],
                    std=[0.268, 0.261, 0.275],
                ),
            ])

        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.481, 0.457, 0.408],
                std=[0.268, 0.261, 0.275],
            ),
        ])

    def get_au_list(self) -> List[int]:
        """Return the AU order in the label vector."""
        return list(self.au_ids)

    def _resolve_image_path(self, raw_path: str) -> str:
        """Resolve absolute and dataset-root-relative image paths."""
        raw_path = str(raw_path).strip()

        if os.path.isabs(raw_path):
            return os.path.abspath(raw_path)

        if self.root is None:
            return os.path.abspath(raw_path)

        return os.path.abspath(
            os.path.join(self.root, raw_path.lstrip("/\\"))
        )

    @staticmethod
    def _extract_sequence_information(
        image_path: str,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Parse sequence information from a folder such as:

            2F23_10

        Returns:
            subject_id:
                F023

            task_id:
                10

            sequence_folder:
                2F23_10
        """
        sequence_folder = os.path.basename(
            os.path.dirname(image_path)
        )

        match = re.fullmatch(
            r"\d+([FM])(\d{2})_(\d{2})",
            sequence_folder,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        gender = match.group(1).upper()
        subject_number = int(match.group(2))
        task_id = match.group(3)

        subject_id = f"{gender}{subject_number:03d}"

        return subject_id, task_id, sequence_folder

    @staticmethod
    def _extract_frame_index(image_path: str) -> Optional[int]:
        """
        Extract frame index from filenames such as:

            0.jpg
            1001.jpg
        """
        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)

        if stem.isdigit():
            return int(stem)

        numbers = re.findall(r"\d+", stem)

        if not numbers:
            return None

        return int(numbers[-1])

    def _validate_image(
        self,
        image_path: str,
    ) -> Tuple[bool, Optional[str]]:
        if not os.path.isfile(image_path):
            return False, "missing"

        if self.verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                return False, "corrupted"

        return True, None

    def _parse_line(
        self,
        line: str,
        line_number: int,
    ) -> Optional[Dict]:
        # Validate the fixed path-plus-twelve-binary-label manifest schema.
        parts = line.split()
        expected_fields = 1 + self.num_aus

        if len(parts) != expected_fields:
            self.invalid_records.append({
                "line_number": line_number,
                "reason": (
                    f"Expected {expected_fields} fields, got {len(parts)}."
                ),
                "line": line,
            })
            return None

        raw_path = parts[0]
        label_tokens = parts[1:]

        try:
            labels = [int(value) for value in label_tokens]
        except ValueError as error:
            self.invalid_records.append({
                "line_number": line_number,
                "reason": f"Cannot parse labels: {error}",
                "line": line,
            })
            return None

        if any(value not in (0, 1) for value in labels):
            self.invalid_records.append({
                "line_number": line_number,
                "reason": "AU labels must be binary values 0 or 1.",
                "line": line,
            })
            return None

        image_path = self._resolve_image_path(raw_path)

        sequence_information = self._extract_sequence_information(
            image_path
        )

        if sequence_information is None:
            self.invalid_records.append({
                "line_number": line_number,
                "reason": (
                    "Cannot parse sequence folder. Expected a form such as "
                    "2F23_10."
                ),
                "image_path": image_path,
            })
            return None

        subject_id, task_id, sequence_folder = sequence_information

        if subject_id not in self.subject_to_index:
            self.invalid_records.append({
                "line_number": line_number,
                "reason": f"Unknown BP4D subject: {subject_id}.",
                "image_path": image_path,
            })
            return None

        frame_index = self._extract_frame_index(image_path)

        if frame_index is None:
            self.invalid_records.append({
                "line_number": line_number,
                "reason": "Cannot parse frame index from filename.",
                "image_path": image_path,
            })
            return None

        is_valid, invalid_type = self._validate_image(image_path)

        if not is_valid:
            invalid_info = {
                "line_number": line_number,
                "image_path": image_path,
            }

            if invalid_type == "missing":
                self.missing_files.append(invalid_info)
            else:
                self.corrupted_files.append(invalid_info)

            return None

        return {
            "line_number": line_number,
            "image_path": image_path,
            "labels": labels,
            "subject_id": subject_id,
            "task_id": task_id,
            "sequence_folder": sequence_folder,
            "sequence_path": os.path.dirname(image_path),
            "frame_index": frame_index,
        }

    def _load_valid_frames(self) -> Dict[str, List[Dict]]:
        if not os.path.isfile(self.label_file):
            raise FileNotFoundError(
                f"Label file not found: {self.label_file}"
            )

        # Group frames before packing so clips never cross task sequences.
        grouped_frames: Dict[str, List[Dict]] = defaultdict(list)

        with open(self.label_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                self.total_records += 1

                record = self._parse_line(
                    line=line,
                    line_number=line_number,
                )

                if record is None:
                    continue

                grouped_frames[record["sequence_path"]].append(record)
                self.valid_frame_count += 1

        invalid_count = (
            len(self.invalid_records)
            + len(self.missing_files)
            + len(self.corrupted_files)
        )

        if invalid_count > 0 and not self.skip_invalid:
            messages = [
                "BP4D validation failed during dataset initialization.",
                f"Invalid records: {len(self.invalid_records)}",
                f"Missing images: {len(self.missing_files)}",
                f"Corrupted images: {len(self.corrupted_files)}",
            ]

            for item in self.missing_files[:10]:
                messages.append(f"Missing: {item['image_path']}")

            for item in self.corrupted_files[:10]:
                messages.append(f"Corrupted: {item['image_path']}")

            raise RuntimeError("\n".join(messages))

        return grouped_frames

    def _remove_duplicate_frames(
        self,
        sequence_path: str,
        frame_records: List[Dict],
    ) -> List[Dict]:
        """
        Retain only the first record when one sequence contains duplicate frame
        indices.
        """
        unique_records: Dict[int, Dict] = {}

        for record in sorted(
            frame_records,
            key=lambda item: (
                item["frame_index"],
                item["line_number"],
            ),
        ):
            frame_index = record["frame_index"]

            if frame_index in unique_records:
                self.duplicate_frames.append({
                    "sequence_path": sequence_path,
                    "frame_index": frame_index,
                    "kept_path": unique_records[frame_index]["image_path"],
                    "removed_path": record["image_path"],
                })
                continue

            unique_records[frame_index] = record

        return [
            unique_records[index]
            for index in sorted(unique_records)
        ]

    @staticmethod
    def _split_continuous_segments(
        frame_records: List[Dict],
    ) -> List[List[Dict]]:
        """
        Split one sequence at every missing frame.

        Example:
            0, 1, 2, 5, 6

        becomes:
            [0, 1, 2]
            [5, 6]
        """
        if not frame_records:
            return []

        frame_records = sorted(
            frame_records,
            key=lambda item: item["frame_index"],
        )

        segments: List[List[Dict]] = []
        current_segment = [frame_records[0]]

        for current_record in frame_records[1:]:
            previous_index = current_segment[-1]["frame_index"]
            current_index = current_record["frame_index"]

            if current_index == previous_index + 1:
                current_segment.append(current_record)
            else:
                segments.append(current_segment)
                current_segment = [current_record]

        segments.append(current_segment)
        return segments

    def _build_samples(self) -> None:
        grouped_frames = self._load_valid_frames()
        self.sequence_count = len(grouped_frames)

        for sequence_path, frame_records in grouped_frames.items():
            frame_records = self._remove_duplicate_frames(
                sequence_path=sequence_path,
                frame_records=frame_records,
            )

            segments = self._split_continuous_segments(frame_records)
            self.continuous_segment_count += len(segments)

            for segment in segments:
                if len(segment) < self.clip_len:
                    self.short_segment_count += 1
                    continue

                last_start = len(segment) - self.clip_len

                # Use the original non-overlapping 16-frame start positions.
                for start in range(
                    0,
                    last_start + 1,
                    self.stride,
                ):
                    clip_records = segment[
                        start:start + self.clip_len
                    ]

                    subjects = {
                        record["subject_id"]
                        for record in clip_records
                    }

                    sequence_paths = {
                        record["sequence_path"]
                        for record in clip_records
                    }

                    frame_indices = [
                        record["frame_index"]
                        for record in clip_records
                    ]

                    expected_indices = list(
                        range(
                            frame_indices[0],
                            frame_indices[0] + self.clip_len,
                        )
                    )

                    if len(subjects) != 1:
                        continue

                    if len(sequence_paths) != 1:
                        continue

                    if frame_indices != expected_indices:
                        continue

                    self.samples.append(clip_records)

    def _apply_transform_to_clip(
        self,
        images: List[Image.Image],
    ) -> List[torch.Tensor]:
        """
        Apply identical random transform parameters to all frames in one clip.
        """
        if not self.sync_transform:
            return [
                self.transform(image)
                for image in images
            ]

        # Replay one sampled transform seed for every frame in the clip.
        clip_seed = int(
            torch.randint(
                low=0,
                high=2**31 - 1,
                size=(1,),
            ).item()
        )

        # Restore global RNG states so synchronization adds no extra draws.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        transformed_images: List[torch.Tensor] = []

        try:
            for image in images:
                random.seed(clip_seed)
                np.random.seed(clip_seed)
                torch.manual_seed(clip_seed)

                transformed_images.append(
                    self.transform(image)
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

        return transformed_images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return time-major image and AU tensors with matching frame order.
        clip_records = self.samples[index]

        images: List[Image.Image] = []
        labels: List[torch.Tensor] = []

        for record in clip_records:
            image_path = record["image_path"]

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    "Image was removed or moved after dataset initialization: "
                    f"{image_path}"
                )

            images.append(self.loader(image_path))

            labels.append(
                torch.tensor(
                    record["labels"],
                    dtype=torch.float32,
                )
            )

        transformed_images = self._apply_transform_to_clip(images)

        clip = torch.stack(
            transformed_images,
            dim=0,
        )

        au_sequence = torch.stack(
            labels,
            dim=0,
        )

        return clip, au_sequence

    def get_sequence_metadata(self, index: int) -> Dict:
        """
        Return the untransformed paths and labels for one packed clip.
        """
        records = self.samples[index]

        first_record = records[0]

        return {
            "dataset_index": index,
            "subject_id": first_record["subject_id"],
            "task_id": first_record["task_id"],
            "sequence_folder": first_record["sequence_folder"],
            "sequence_path": first_record["sequence_path"],
            "start_frame": records[0]["frame_index"],
            "end_frame": records[-1]["frame_index"],
            "frames": [
                {
                    "position": position,
                    "frame_index": record["frame_index"],
                    "image_path": record["image_path"],
                    "labels": list(record["labels"]),
                    "active_aus": [
                        au
                        for au, value in zip(
                            self.au_ids,
                            record["labels"],
                        )
                        if value == 1
                    ],
                }
                for position, record in enumerate(records)
            ],
        }

    def _print_summary(self) -> None:
        print("=" * 72)
        print("BP4D 16-frame dataset summary")
        print(f"Label file:              {self.label_file}")
        print(f"Image root:              {self.root}")
        print(f"Label records:           {self.total_records}")
        print(f"Valid image frames:      {self.valid_frame_count}")
        print(f"Sequence folders:        {self.sequence_count}")
        print(f"Missing images:          {len(self.missing_files)}")
        print(f"Corrupted images:        {len(self.corrupted_files)}")
        print(f"Invalid records:         {len(self.invalid_records)}")
        print(f"Duplicate frames:        {len(self.duplicate_frames)}")
        print(f"Continuous segments:     {self.continuous_segment_count}")
        print(f"Short segments:          {self.short_segment_count}")
        print(f"Clip length:             {self.clip_len}")
        print(f"Clip stride:             {self.stride}")
        print(f"Valid clips:             {len(self.samples)}")

        if self.missing_files:
            print("-" * 72)
            print("Missing image examples:")

            for item in self.missing_files[:20]:
                print(f"  {item['image_path']}")

            remaining = len(self.missing_files) - 20

            if remaining > 0:
                print(f"  ... and {remaining} more")

        if self.invalid_records:
            print("-" * 72)
            print("Invalid record examples:")

            for item in self.invalid_records[:10]:
                print(
                    f"  line {item.get('line_number')}: "
                    f"{item.get('reason')}"
                )

        print("=" * 72)
