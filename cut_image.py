import numpy as np
from PIL import Image
from cut_picture import start_points
import argparse
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm


def crop_image(output_image_dir, image_path, crop_size, stride):
    # 图像名称
    img_id = image_path.stem

    # 加载图像和标签
    image = np.asarray(Image.open(image_path))

    # 通过设置crop_size和stride，可以控制重叠比例
    width, height = image.shape[:2]
    X_points = start_points(width, crop_size, stride=stride)
    Y_points = start_points(height, crop_size, stride=stride)

    count = 1
    for i in X_points:
        for j in Y_points:
            new_image = image[i:i + crop_size, j:j + crop_size]
            new_image = Image.fromarray(new_image)
            new_image.save("{}/{}_{}.png".format(output_image_dir, img_id, count))
            count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop Remote Sense Datasets")
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--input_dir', default=Path(r'/media/DiskDP/datasets/YqDataset/Wayback/train/image'))
    parser.add_argument('--output_dir', default=Path(r'/media/DiskDP/datasets/YqDataset/cutImage/Wayback'))
    args = parser.parse_args()

    # 设置输出路径，并判断路径是否存在，不存在就创建
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 获取图像和mask列表，并排序
    image_list = natsorted(list(args.input_dir.glob('*.png')))[:7000]

    # 对每张图像和mask进行裁剪，并保存到temp文件夹下
    for idx in tqdm(range(len(image_list)), desc='cropping dtasets images'):
        crop_image(args.output_dir, image_list[idx], args.crop_size, args.stride)
