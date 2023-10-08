from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import mask_deal, save_colored_mask

if __name__ == '__main__':
    # 定义颜色映射规则
    label_map = {
        (0, 128, 0): 1,  # 林地
        (128, 0, 0): 2,  # 种植地
        (96, 0, 0): 3,  # 水体
        (0, 0, 255): 4,  # 建筑物
        (0, 64, 0): 5,  # 桥梁
        (128, 0, 128): 5,  # 道路
        (191, 0, 0): 6,  # 硬化地表
        (0, 128, 128): 7,  # 人工裸地
        (128, 128, 128): 7,  # 自然裸地
        (128, 128, 0): 8,  # 草地
        (0, 0, 0): 0,  # 未识别
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
    yq_map = [
        (255, 0, 0),  # 红色 未识别
        (0, 201, 87),  # 绿色林地
        (255, 255, 0),  # 黄色种植地
        (0, 191, 255),  # 蓝色水体
        (255, 255, 255),  # 白色建筑
        (0, 0, 0),  # 黑色道路
        (135, 240, 225),  # 青色人工构建
        (128, 0, 128),  # 紫色裸地
        (50, 50, 50),  # 深灰色草地
        (105, 105, 105)  # 浅灰色 其他
    ]

    # 设置输入和输出文件夹路径
    input_folder = Path(r'C:\Users\eveLe\Downloads\process_data\mask')
    output_folder = Path(r'C:\Users\eveLe\Downloads\process_data\temp')

    # 确保输出文件夹存在
    output_folder.mkdir(parents=True, exist_ok=True)

    mask_list = list(input_folder.glob('*'))

    for mask in tqdm(mask_list):
        mask_data = np.asarray(Image.open(mask))
        new_mask = mask_deal(mask_data, label_map, background=9)
        save_path = output_folder.joinpath(mask.name)
        save_colored_mask(new_mask, save_path, al_map)

    print("所有图片处理完成！")
