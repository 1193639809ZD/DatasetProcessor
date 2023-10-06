from PIL import Image
from tqdm import tqdm
import os

# 禁用解压缩炸弹限制
Image.MAX_IMAGE_PIXELS = None

# 定义输入和输出文件夹路径
input_folder = r'cache\1'
output_folder = r'cache\2'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片文件的路径
image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if
               filename.endswith('.png') or filename.endswith('.jpg')]

# 遍历图片文件
for i, image_path in tqdm(enumerate(image_paths)):
    # 打开大图
    image = Image.open(image_path)

    # 获取大图的宽度和高度
    width, height = image.size

    # 计算裁剪后的小图数量（行数和列数）
    num_rows = height // 1024
    num_cols = width // 1024

    # 遍历裁剪位置
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算裁剪的起始点坐标和结束点坐标
            left = col * 1024
            upper = row * 1024
            right = left + 1024
            lower = upper + 1024

            # 裁剪小图
            cropped_image = image.crop((left, upper, right, lower))

            # 构造小图的保存路径
            output_path = os.path.join(output_folder, f"{i}_r{row}_c{col}.png")

            # 保存小图
            cropped_image.save(output_path)

print("所有图片裁剪完成！")
