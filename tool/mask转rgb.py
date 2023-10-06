from PIL import Image
import os

# 定义颜色映射规则
color_mapping = {
    0: (0, 0, 0),  # 黑色
    1: (0, 201, 87),  # 绿色
    2: (255, 255, 0),  # 黄色
    3: (0, 191, 255),  # 蓝色
    4: (255, 255, 255),  # 白色
    5: (128, 0, 128),  # 紫色
    6: (128, 128, 128)  # 灰色
}

# 设置输入和输出文件夹路径
input_folder = r'cache\longan\4'
output_folder = r'cache\longan\5'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 构造单通道图像的完整路径
        single_channel_image_path = os.path.join(input_folder, filename)

        # 打开单通道图像
        single_channel_image = Image.open(single_channel_image_path)

        # 创建彩色图像
        colored_image = Image.new('RGB', single_channel_image.size)

        # 遍历每个像素点，并根据颜色映射规则染色
        for y in range(single_channel_image.height):
            for x in range(single_channel_image.width):
                # 获取单通道图像的像素值
                value = single_channel_image.getpixel((x, y))
                # 根据颜色映射规则获取颜色值
                color = color_mapping.get(value, (0, 0, 0))  # 默认设置为黑色
                # 设置彩色图像对应像素的值
                colored_image.putpixel((x, y), color)

        # 构造输出图像的完整路径
        output_image_path = os.path.join(output_folder, filename)

        # 保存彩色图像
        colored_image.save(output_image_path)

print("所有图片处理完成！")
