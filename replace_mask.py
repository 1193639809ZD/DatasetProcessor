from PIL import Image
from pathlib import Path
import numpy as np
from natsort import natsorted
from utils import save_colored_mask
from tqdm import tqdm

if __name__ == '__main__':
    # target mask and origin mask are palette mode
    target_list = natsorted(list(Path(r'D:\datasets\yq-tech\DeepGlobe\origin\platte-mask').glob('*.png')))
    mask_list = natsorted(list(Path(r'D:\datasets\yq-tech\DeepGlobe\origin\mask').glob('*.png')))
    output_dir = Path(r'D:\datasets\yq-tech\DeepGlobe\origin\temp')
    # 阿里标签
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
    # assert len(target_list) == len(mask_list)
    replace_label = 5
    target_label = 1
    for idx in tqdm(range(len(target_list))):
        target_mask = np.asarray(Image.open(target_list[idx]))
        mask = np.asarray(Image.open(mask_list[idx])).copy()
        # 消除mask的原目标标签
        locations = mask == replace_label
        mask[locations] = 0
        # 写入目标mask的标签
        locations = target_mask == target_label
        mask[locations] = replace_label
        save_colored_mask(mask, output_dir.joinpath(f"{idx + 1}.png"), al_map)
