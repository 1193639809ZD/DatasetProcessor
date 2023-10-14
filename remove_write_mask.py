"""
功能：去除图像空白区域的mask
"""

from pathlib import Path
from natsort import natsorted
from utils import cut_mask
from tqdm import tqdm

if __name__ == '__main__':
    image_dir = Path(r'D:\datasets\Massachusetts_Dataset\image')
    mask_dir = Path(r'D:\datasets\Massachusetts_Dataset\al_mask')
    output = Path(r'D:\datasets\Massachusetts_Dataset\new')

    image_list = natsorted(list(image_dir.glob('*.png')))
    mask_list = natsorted(list(mask_dir.glob('*.png')))
    assert len(image_list) == len(mask_list)

    for idx in tqdm(range(len(image_list))):
        output_path = output.joinpath(mask_list[idx].name)
        cut_mask(image_list[idx], mask_list[idx], output_path)
