from PIL import Image
import os

# 设置输入和输出文件夹路径
input_folder = 'C:\\Users\DELL\Desktop\\akesu\\'
output_folder = 'C:\\Users\DELL\Desktop\\sj\\1\\'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有png图片文件，并按文件名排序
image_files = sorted([
    filename for filename in os.listdir(input_folder) if filename.endswith('.png')
])

# 定义拼接后的大图尺寸和行列数
big_image_size = 1024
tile_size = 256
tiles_per_row = big_image_size // tile_size

# 初始化拼接后的大图和计数器
big_image = Image.new('RGB', (big_image_size, big_image_size))
counter = 1

# 遍历每一组16张图片并拼接
for i in range(0, len(image_files), 16):
    image_group = image_files[i:i + 16]

    # 创建一个新的空白大图
    big_image = Image.new('RGB', (big_image_size, big_image_size))

    # 遍历当前组中的每张图片
    for j, image_name in enumerate(image_group):
        image_path = os.path.join(input_folder, image_name)

        # 打开单张图片
        image = Image.open(image_path)

        # 计算当前图片在大图中的位置
        row = j // tiles_per_row
        col = j % tiles_per_row

        # 计算当前图片的左上角坐标
        x = col * tile_size
        y = row * tile_size

        # 将当前图片拷贝到大图中的相应位置
        big_image.paste(image, (x, y))

    # 拼接后的大图文件路径
    output_image_path = os.path.join(output_folder, str(counter) + '.png')

    # 保存拼接后的大图
    big_image.save(output_image_path)

    counter += 1

print("所有图片拼接完成！")
