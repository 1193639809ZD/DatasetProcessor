from __future__ import print_function, division
import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import random
from torchvision.transforms import ToPILImage


# 测试裁剪缩放对标签的影像

class RandomResizedCrop(object):
    def __init__(self, size=256, scale=(0.3, 1.0), probability=1.0):
        self.trans = transforms.RandomResizedCrop(size, scale=scale, interpolation=InterpolationMode.NEAREST)
        self.probability = probability

    def __call__(self, sample):
        img_tensor, map_img_tensor = sample["img"], sample["map_img"]

        if random.random() < self.probability:
            # map_img_tensor = map_img_tensor.unsqueeze(0)
            x = torch.cat([img_tensor, map_img_tensor], dim=0)
            x = self.trans(x)
            img_tensor, map_img_tensor = x[0:3, ...], x[3, ...].unsqueeze(0)

        return {
            "img": img_tensor,
            "map_img": map_img_tensor
        }


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.trans = transforms.RandomHorizontalFlip(prob)

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]
        x = torch.cat([img, map_img], dim=0)
        x = self.trans(x)

        return {
            "img": x[0:3, ...],
            "map_img": x[3, ...].unsqueeze(0)
        }


class RandomRotation(object):
    def __init__(self, angle=45, probability=1.0):
        self.trans = transforms.RandomRotation(angle)
        self.probability = probability

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]

        if random.random() < self.probability:
            x = torch.cat([img, map_img], dim=0)
            x = self.trans(x)
            img, map_img = x[0:3, ...], x[3, ...].unsqueeze(0)

        return {
            "img": img,
            "map_img": map_img
        }


# 颜色扰动
class ColorJitter(object):
    def __init__(self, l=0.3, d=0.3, b=0.3, s=0.3, probability=1.0):
        self.trans = transforms.ColorJitter(l, d, b, s)
        self.probability = probability

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]

        if random.random() < self.probability:
            img = self.trans(img)

        return {
            "img": img,
            "map_img": map_img
        }


transform = transforms.Compose([
    RandomResizedCrop(size=256, scale=(0.3, 1.0), probability=0.5),
    RandomHorizontalFlip(prob=0.5),
    RandomRotation(angle=45, probability=0.5),
    ColorJitter(l=0.3, d=0.3, b=0.3, s=0.3, probability=0.5),
])


def enhance(batch):
    images = batch["img"]
    labels = batch["map_img"]

    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        sample = {"img": images[i], "map_img": labels[i]}
        augmented_sample = transform(sample)

        augmented_images.append(augmented_sample["img"])
        augmented_labels.append(augmented_sample["map_img"])

    augmented_batch = {
        "img": torch.stack(augmented_images),
        "map_img": torch.stack(augmented_labels)
    }

    return augmented_batch
