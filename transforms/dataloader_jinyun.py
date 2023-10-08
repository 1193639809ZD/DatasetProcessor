"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division

import numpy
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import torch
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, hp, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.path = hp.train if train else hp.valid
        self.mask_list = glob.glob(
            os.path.join(self.path, "jinyun_jingning_mask", "*.png"), recursive=True  # 土拨鼠更新文件名和后缀
        )
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        mask = Image.open(maskpath)
        image = Image.open(maskpath.replace("jinyun_jingning_mask", "jinyun_jingning")).convert('RGB')

        sample = {"img": image, "map_img": mask}

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomResizedCrop(object):
    def __init__(self, size=256, scale=(0.3, 1.0)):
        self.trans = transforms.RandomResizedCrop(size, scale=scale)

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]
        x = torch.cat([img, map_img], dim=0)
        x = self.trans(x)
        return {
            "img": x[0:3, ...],
            "map_img": x[3, ...].unsqueeze(0)
        }  # unsqueeze for the channel dimension


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
        }  # unsqueeze for the channel dimension


class RandomRotation(object):
    def __init__(self, angle=45):
        self.trans = transforms.RandomRotation(angle)

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]
        x = torch.cat([img, map_img], dim=0)
        x = self.trans(x)
        return {
            "img": x[0:3, ...],
            "map_img": x[3, ...].unsqueeze(0)
        }  # unsqueeze for the channel dimension


class ColorJitter(object):
    def __init__(self, l=0.3, d=0.3, b=0.3, s=0.3):
        self.trans = transforms.ColorJitter(l, d, b, s)

    def __call__(self, sample):
        img, map_img = sample["img"], sample["map_img"]
        x = self.trans(img)
        return {
            "img": x,
            "map_img": map_img
        }  # unsqueeze for the channel dimension


class RandomGrayscale(object):
    def __init__(self, prob=0.2):
        self.trans = transforms.RandomGrayscale(p=prob)

    def __call__(self, sample):
        image_before, image_after, map_img = sample["img_before"], sample["img_after"], sample["map_img"]
        x = self.trans(image_before)
        y = self.trans(image_after)

        return {
            "img_before": x,
            "img_after": y,
            "map_img": map_img
        }  # unsqueeze for the channel dimension


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, mask = sample["img"], sample["map_img"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            "img": transforms.functional.to_tensor(img),
            "map_img": transforms.functional.to_tensor(mask).float()
        }  # unsqueeze for the channel dimension


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        image_before, image_after, map_img = sample["img_before"], sample["img_after"], sample["map_img"]
        x = transforms.Normalize(self.mean[:3], self.std[:3])(torch.from_numpy(image_before).permute(2, 0, 1).float())
        y = transforms.Normalize(self.mean[3:], self.std[3:])(torch.from_numpy(image_after).permute(2, 0, 1).float())
        return {
            "img_before": x,
            "img_after": y,
            "map_img": torch.from_numpy(map_img).unsqueeze(0).float()
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
