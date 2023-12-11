from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import imgviz
import random


# 切除图像空白区域的mask标签
def cut_mask(image, mask, output, color_map):
    # 读取数据
    img_data = np.asarray(Image.open(image))
    mask_data = np.asarray(Image.open(mask)).copy()
    # 修改mask
    locations = img_data == [255, 255, 255]
    locations = np.all(locations, axis=-1)
    mask_data[locations] = 0
    # mask_data[locations] = [0, 0, 0]
    # 保存
    save_colored_mask(mask_data, output, color_map)
    # mask_i = Image.fromarray(mask_data)
    # mask_i.save(output)


def to_png(image_dir, output_dir):
    """
    功能：文件夹的图像转成png格式

    :param image_dir:
    :param output_dir:
    :return:
    """
    data = Path(image_dir)
    output = Path(output_dir)
    image_list = list(data.glob('*'))

    # 将图像转化为png格式
    for image in tqdm(image_list):
        im_data = np.asarray(Image.open(image))
        img = Image.fromarray(im_data)
        img.save(output.joinpath(image.name).with_suffix('.png'))


def save_colored_mask(mask, save_path, palette_map=None):
    """
    功能：以调色板模式保存mask

    :param mask: mask数据，numpy格式
    :param save_path: 保存路径，str或者Path格式
    :param palette_map: 可选，指定调色板颜色
    :return: 无返回
    """
    mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    label_colormap = imgviz.label_colormap()
    # 若指定了调色板则进行替换
    if palette_map is not None:
        for idx in range(len(palette_map)):
            label_colormap[idx] = palette_map[idx]
    mask.putpalette(label_colormap.flatten())
    mask.save(save_path)


def mask_deal(mask_data, color_mapping, background=0):
    """
    功能：将mask映射到标签空间

    :param mask_data: mask数据，为numpy格式
    :param color_mapping: 颜色映射关系，为字典格式，比如{(0, 128, 0): 1, (128, 0, 0): 2}
    :param background: 背景，或者默认标签
    :return: 处理后的mask数据，为numpy格式
    """
    new_mask = np.full((mask_data.shape[0], mask_data.shape[1]), background, dtype=np.uint8)
    for color in color_mapping:
        locations = mask_data == color
        if len(locations.shape) == 3:
            locations = np.all(locations, axis=-1)
        new_mask[locations] = color_mapping[color]
    return new_mask


def list_split(full_list, ratio):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     划分比率（ratio：1-ratio）
    """
    offset = int(len(full_list) * ratio)
    random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
