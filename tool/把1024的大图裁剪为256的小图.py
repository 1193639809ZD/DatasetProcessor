from PIL import Image
import os

# 定义输入和输出文件夹路径
input_folder = 'C:\\Users\DELL\Desktop\sj\\2\\'
output_folder = 'C:\\Users\DELL\Desktop\sj\\6\\'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片文件的路径
image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.png') or filename.endswith('.jpg')]

# 遍历图片文件
for i, image_path in enumerate(image_paths):
    # 打开大图
    image = Image.open(image_path)

    # 获取大图的宽度和高度
    width, height = image.size

    # 确保大图大小是1024x1024
    if width == 1024 and height == 1024:
        # 遍历裁剪位置
        for row in range(4):
            for col in range(4):
                # 计算裁剪的起始点坐标和结束点坐标
                left = col * 256
                upper = row * 256
                right = left + 256
                lower = upper + 256

                # 裁剪小图
                cropped_image = image.crop((left, upper, right, lower))

                # 构造小图的保存路径
                output_path = os.path.join(output_folder, f"{i*16 + row*4 + col + 1}.png")

                # 保存小图
                cropped_image.save(output_path)

print("所有图片裁剪完成！")
