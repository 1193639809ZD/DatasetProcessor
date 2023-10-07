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
    parse.add_argument('-d', '--dir', type=str, default=r'D:\datasets\datas\mask')
    parse.add_argument('-s', '--suffix', type=str, default='png', help="mask文件后缀")
    args = parse.parse_args()

    # label_map = {
    #     255: 5,  # 道路
    #     0: 0  # 背景
    # }

    output = Path(r'D:\datasets\datas\temp')
    # 打开文件夹
    root = Path(args.dir)
    mask_list = list(root.glob(f'*.{args.suffix}'))
    # 遍历每个文件，并转为调色板模式
    for mask_path in tqdm(mask_list):
        mask = np.asarray(Image.open(mask_path))
        # if label_map is not None:
        #     mask = mask_deal(mask, label_map)
        save_colored_mask(mask, output.joinpath(mask_path.name))
