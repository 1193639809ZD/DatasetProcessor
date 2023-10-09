import argparse
from pathlib import Path

import imgviz
import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm


def load_image(file_path):
    img = Image.open(file_path)
    data = np.asarray(img, dtype="int32")
    return data


def save_colored_mask(mask, save_path):
    mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    mask.putpalette(colormap.flatten())
    mask.save(save_path)


# 返回坐标轴切割的起点坐标
def start_points(size, patch_size, stride=256):
    points = [0]
    counter = 1
    while True:
        pt = stride * counter
        if pt + patch_size >= size:
            points.append(size - patch_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def crop_image_mask(image_dir, mask_dir, image_path, mask_path, crop_size, stride):
    # 图像名称
    img_id = mask_path.stem

    # 加载图像和标签
    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path))

    # 通过设置crop_size和stride，可以控制重叠比例
    width, height = mask.shape
    X_points = start_points(width, crop_size, stride=stride)
    Y_points = start_points(height, crop_size, stride=stride)

    count = 1
    for i in X_points:
        for j in Y_points:
            new_image = image[i:i + crop_size, j:j + crop_size]
            new_mask = mask[i:i + crop_size, j:j + crop_size]
            # Skip any Image that is more than 99% empty.
            if np.any(new_mask):
                # 返回背景和前景的数量，如果背景太多略过
                num_clas_pixel = np.unique(new_mask, return_counts=True)[1]
                if num_clas_pixel[0] / num_clas_pixel.sum() > 0.99:
                    continue
                # 保存图片，mask以调色板模式保存
                save_colored_mask(new_mask, mask_dir.joinpath(f'{img_id}_{count}.png'))
                new_image = Image.fromarray(new_image)
                new_image.save("{}/{}_{}.png".format(image_dir, img_id, count))
                count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop Remote Sense Datasets")
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--dataset_root', default=Path(r'D:\datasets\yq-tech\DeepGlobe'))
    args = parser.parse_args()

    # 设置输出路径，并判断路径是否存在，不存在就创建
    temp_image_dir = args.dataset_root.joinpath(r'temp\image')
    temp_mask_dir = args.dataset_root.joinpath(r'temp\mask')
    temp_image_dir.mkdir(parents=True, exist_ok=True)
    temp_mask_dir.mkdir(parents=True, exist_ok=True)

    # 获取图像和mask列表，并排序
    image_list = natsorted(list(args.dataset_root.glob('image\*.png')))
    mask_list = natsorted(list(args.dataset_root.glob('mask\*.png')))
    print("Length of image :", len(image_list))
    print("Length of mask :", len(mask_list))
    assert len(image_list) == len(mask_list)

    # 对每张图像和mask进行裁剪，并保存到temp文件夹下
    for idx in tqdm(range(len(mask_list)), desc='cropping dtasets images'):
        crop_image_mask(temp_image_dir, temp_mask_dir, image_list[idx], mask_list[idx], args.crop_size, args.stride)
