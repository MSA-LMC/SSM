import math
import numbers
import random

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageOps


# Group transforms sample one spatial decision and apply it to every frame.
class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        out_images = []
        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images


class GroupRandomHorizontalFlip:
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        # A single RNG draw keeps the full clip temporally aligned.
        if random.random() < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        return img_group


class GroupScale:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupResize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        out_group = []
        for img in img_group:
            out_group.append(img.resize((self.size, self.size), self.interpolation))
        return out_group


class GroupRandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        # Match the ten-attempt random-resized-crop fallback from DFER-CLIP.
        for _ in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = []
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group

        scale = GroupScale(self.size, interpolation=self.interpolation)
        crop = GroupRandomCrop(self.size)
        return crop(scale(img_group))


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None
        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None
        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            raise TypeError("Color jitter is not implemented for numpy arrays")
        if not isinstance(clip[0], PIL.Image.Image):
            raise TypeError(
                f"Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}"
            )

        # Reuse one sampled jitter recipe across all frames in the clip.
        brightness, contrast, saturation, hue = self.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
        )
        img_transforms = []
        if brightness is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_brightness(
                    img, brightness
                )
            )
        if saturation is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_saturation(
                    img, saturation
                )
            )
        if hue is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_hue(img, hue)
            )
        if contrast is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_contrast(
                    img, contrast
                )
            )
        random.shuffle(img_transforms)

        jittered_clip = []
        for img in clip:
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(jittered_img)
        return jittered_clip


class RandomRotation:
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("degrees must be positive")
            degrees = (-degrees, degrees)
        elif len(degrees) != 2:
            raise ValueError("degrees must contain two values")
        self.degrees = degrees

    def __call__(self, clip):
        # FERV39K applies one shared rotation angle to the complete clip.
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            exit()
        if isinstance(clip[0], PIL.Image.Image):
            return [img.rotate(angle) for img in clip]
        raise TypeError(
            f"Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}"
        )


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        # Concatenate frames along channels before conversion to a tensor.
        if img_group[0].mode in ("L", "F"):
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        if img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            return np.concatenate(img_group, axis=2)
        return None


class ToTorchFormatTensor:
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = list(mean)
        self.std = list(std)
        self.base_c = len(self.mean)

    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)

        dtype = tensor.dtype
        device = tensor.device

        def make_mean_std(channels, expand_dims):
            # Repeat RGB statistics for channel-stacked video frames.
            if channels == self.base_c:
                mean = torch.tensor(self.mean, dtype=dtype, device=device)
                std = torch.tensor(self.std, dtype=dtype, device=device)
            elif channels % self.base_c == 0:
                repeats = channels // self.base_c
                mean = torch.tensor(self.mean, dtype=dtype, device=device).repeat(
                    repeats
                )
                std = torch.tensor(self.std, dtype=dtype, device=device).repeat(repeats)
            else:
                raise RuntimeError(
                    f"Channel size {channels} is not compatible with "
                    f"base C={self.base_c}"
                )
            return mean.view(*expand_dims), std.view(*expand_dims)

        if tensor.dim() == 3:
            c = tensor.size(0)
            mean, std = make_mean_std(c, (-1, 1, 1))
            return (tensor - mean) / std

        if tensor.dim() == 4:
            b, c, h, w = tensor.size()
            if c % self.base_c == 0:
                mean, std = make_mean_std(c, (1, -1, 1, 1))
                return (tensor - mean) / std
            if b % 1 == 0 and tensor.size(1) == self.base_c:
                mean = torch.tensor(self.mean, dtype=dtype, device=device).view(
                    1, -1, 1, 1
                )
                std = torch.tensor(self.std, dtype=dtype, device=device).view(
                    1, -1, 1, 1
                )
                return (tensor - mean) / std

        if tensor.dim() == 5:
            mean = torch.tensor(self.mean, dtype=dtype, device=device).view(
                1, 1, -1, 1, 1
            )
            std = torch.tensor(self.std, dtype=dtype, device=device).view(
                1, 1, -1, 1, 1
            )
            return (tensor - mean) / std

        raise RuntimeError(
            f"Unsupported tensor shape {tuple(tensor.shape)} in GroupNormalize"
        )
