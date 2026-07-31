import glob
import os

import numpy as np
import torchvision
from numpy.random import randint
from PIL import Image
from torch.utils import data

from .group_transforms import (
    ColorJitter,
    GroupNormalize,
    GroupRandomHorizontalFlip,
    GroupRandomSizedCrop,
    GroupResize,
    RandomRotation,
    Stack,
    ToTorchFormatTensor,
)


# Keep the normalization constants used by the original dataset branches.
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = [0.481, 0.457, 0.408]
CLIP_STD = [0.268, 0.261, 0.275]


class VideoRecord:
    # Each manifest row stores a frame directory, frame count, and class ID.
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class EmotionVideoDataset(data.Dataset):
    def __init__(
        self,
        list_file,
        num_segments,
        duration,
        mode,
        transform,
        image_size,
        root=None,
    ):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.root = root
        self._parse_list()

    def _parse_list(self):
        rows = [line.strip().split(" ") for line in open(self.list_file)]
        rows = [item for item in rows]
        if self.root is not None:
            for item in rows:
                if not os.path.isabs(item[0]):
                    item[0] = os.path.join(self.root, item[0])
        self.video_list = [VideoRecord(item) for item in rows]
        print("video number:%d" % len(self.video_list))

    def _get_train_indices(self, record):
        # Preserve the legacy segment sampler and its NumPy RNG call order.
        average_duration = (
            record.num_frames - self.duration + 1
        ) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration
            ) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(
                randint(
                    record.num_frames - self.duration + 1,
                    size=self.num_segments,
                )
            )
        else:
            offsets = np.pad(
                np.array(list(range(record.num_frames))),
                (0, self.num_segments - record.num_frames),
                "edge",
            )
        return offsets

    def _get_test_indices(self, record):
        # Evaluation samples deterministic temporal segment centers.
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (
                record.num_frames - self.duration + 1
            ) / float(self.num_segments)
            offsets = np.array(
                [
                    int(tick / 2.0 + tick * x)
                    for x in range(self.num_segments)
                ]
            )
        else:
            offsets = np.pad(
                np.array(list(range(record.num_frames))),
                (0, self.num_segments - record.num_frames),
                "edge",
            )
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == "train":
            segment_indices = self._get_train_indices(record)
        elif self.mode == "test":
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Lexicographic frame ordering matches the original DFER loader.
        video_frames_path = glob.glob(os.path.join(record.path, "*.jpg"))
        video_frames_path.sort()

        if len(video_frames_path) == 0:
            print(f"No frames found in path: {record.path}")
            return None, record.label

        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.duration):
                if p >= len(video_frames_path):
                    p = len(video_frames_path) - 1
                img = Image.open(video_frames_path[p]).convert("RGB")
                images.append(img)
                if p < record.num_frames - 1:
                    p += 1

        total_needed = self.num_segments * self.duration
        if len(images) < total_needed:
            # Repeat the final frame only when a sequence is unexpectedly short.
            last_img = images[-1]
            images.extend([last_img] * (total_needed - len(images)))

        images = self.transform(images)
        images = images.view(
            self.num_segments * self.duration,
            3,
            self.image_size,
            self.image_size,
        )
        return images, record.label

    def __len__(self):
        return len(self.video_list)


def build_emotion_training_dataset(
    list_file,
    num_segments,
    duration,
    image_size,
    args,
    root=None,
):
    # Augmentation recipes remain dataset-specific for exact reproduction.
    if args.dataset == "DFEW":
        train_transforms = torchvision.transforms.Compose(
            [
                ColorJitter(brightness=0.5),
                GroupRandomSizedCrop(image_size),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
    elif args.dataset == "FERV39K":
        train_transforms = torchvision.transforms.Compose(
            [
                RandomRotation(4),
                GroupRandomSizedCrop(image_size),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
    elif args.dataset == "MAFW":
        train_transforms = torchvision.transforms.Compose(
            [
                GroupRandomSizedCrop(image_size),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported emotion dataset: {args.dataset}")

    return EmotionVideoDataset(
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="train",
        transform=train_transforms,
        image_size=image_size,
        root=root,
    )


def build_emotion_evaluation_dataset(
    list_file,
    num_segments,
    duration,
    image_size,
    root=None,
):
    # All expression benchmarks share the released evaluation transform.
    test_transform = torchvision.transforms.Compose(
        [
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return EmotionVideoDataset(
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="test",
        transform=test_transform,
        image_size=image_size,
        root=root,
    )
