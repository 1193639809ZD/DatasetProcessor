"""
去除阿里透明通道
"""

from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    root = Path(r'C:\Users\eveLe\Downloads\process_data\mask')
    mask_list = list(root.glob('*'))
    for mask in tqdm(mask_list):
        mask_data = np.asarray(Image.open(mask))[:, :, :3]
        new_mask = Image.fromarray(mask_data)
        new_mask.save(mask)
