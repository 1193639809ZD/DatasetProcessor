from PIL import Image
import os

# 定义颜色映射规则
color_mapping = {
    (0, 128, 0): 1,
    (0, 0, 0): 0,
    (128, 0, 0): 2,
    (96, 0, 0): 3,
    (0, 0, 255): 4,
    (0, 64, 0): 5,
    (128, 0, 128): 5,
    (191, 0, 0): 5,
    (0, 128, 128): 5,
    (128, 128, 0): 6,
}

# 设置输入和输出文件夹路径
input_folder = r'cache\longan\3'
output_folder = r'cache\longan\4'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 构造输入图像的完整路径
        input_image_path = os.path.join(input_folder, filename)

        # 打开RGB图像
        image = Image.open(input_image_path)

        # 将图像转换为灰度图像
        gray_image = image.convert('L')

        # 创建单通道图像
        single_channel_image = Image.new('L', gray_image.size)

        # 遍历每个像素点，并根据颜色映射规则修改像素值
        for y in range(image.height):
            for x in range(image.width):
                # 获取RGB值
                rgb_value = image.getpixel((x, y))[:3]  # 排除透明度通道
                # 根据颜色映射规则获取映射后的值
                mapped_value = color_mapping.get(rgb_value, 0)  # 默认设置为0
                # 设置单通道图像对应像素的值
                single_channel_image.putpixel((x, y), mapped_value)

        # 构造输出图像的完整路径
        output_image_path = os.path.join(output_folder, filename)

        # 保存单通道图像
        single_channel_image.save(output_image_path)

print("所有图片处理完成！")
