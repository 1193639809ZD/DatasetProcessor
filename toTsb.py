import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import mask_deal, save_colored_mask

"""
本文件的功能：将一个文件夹中的图像转为调色板模式
"""

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="将文件夹中的mask转为调色板模式")
    parse.add_argument('-d', '--dir', type=str, default=r'D:\datasets\yq-tech\Wayback\temp\mask')
    parse.add_argument('-s', '--suffix', type=str, default='png', help="mask文件后缀")
    args = parse.parse_args()

    label_map = {
        (255, 255, 255): 1,  # 道路
        (0, 0, 0): 0  # 背景
    }
    # 指定调色板颜色
    al_map = [
        (0, 0, 0),  # 黑色 未识别
        (0, 128, 0),  # 林地
        (128, 0, 0),  # 种植地
        (96, 0, 0),  # 水体
        (0, 0, 255),  # 建筑物
        (128, 0, 128),  # 道路
        (191, 0, 0),  # 硬化地表
        (128, 128, 128),  # 裸地
        (128, 128, 0),  # 草地
        (105, 105, 105)  # 浅灰色 其他
    ]
    output = Path(r'D:\datasets\yq-tech\Wayback\temp\temp')
    # 打开文件夹
    root = Path(args.dir)
    mask_list = list(root.glob(f'*.{args.suffix}'))
    # 遍历每个文件，并转为调色板模式
    for mask_path in tqdm(mask_list):
        mask = np.asarray(Image.open(mask_path))
        # if label_map is not None:
        #     mask = mask_deal(mask, label_map)
        save_colored_mask(mask, output.joinpath(mask_path.name), al_map)
