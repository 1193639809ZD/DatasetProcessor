from pathlib import Path

from natsort import natsorted
from tqdm import tqdm

"""
代码功能：把指定文件夹的文件按自然排序进行重命名并改成png格式
"""
if __name__ == '__main__':
    # 获取文件列表
    root = Path(r'D:\datasets\Massachusetts_Dataset\origin\temp')
    file_list = natsorted(list(root.glob('*')))
    # 起始索引
    start_idx = 1101
    for file in tqdm(file_list):
        image_path = root.joinpath(f'{start_idx}.png')
        file.replace(image_path)
        start_idx = start_idx + 1
