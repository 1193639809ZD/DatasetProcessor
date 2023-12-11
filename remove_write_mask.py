"""
功能：去除图像空白区域的mask
"""

from pathlib import Path
from natsort import natsorted
from utils import cut_mask
from tqdm import tqdm

if __name__ == '__main__':
    image_dir = Path(r'D:\datasets\Massachusetts_Dataset\image')
    mask_dir = Path(r'D:\datasets\Massachusetts_Dataset\mask')
    output = Path(r'D:\datasets\Massachusetts_Dataset\temp')

    image_list = natsorted(list(image_dir.glob('*.png')))
    mask_list = natsorted(list(mask_dir.glob('*.png')))
    assert len(image_list) == len(mask_list)

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

    for idx in tqdm(range(len(image_list))):
        output_path = output.joinpath(mask_list[idx].name)
        cut_mask(image_list[idx], mask_list[idx], output_path, al_map)
