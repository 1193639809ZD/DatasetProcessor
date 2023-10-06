import os

# 定义文件夹路径
folder_path = r'D:\Project\python-download-tile\cache\longan\3'

# 获取文件夹中所有图片文件的路径
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
               filename.endswith('.png') or filename.endswith('.jpg')]

# 遍历图片文件并重命名
for i, image_path in enumerate(image_paths):
    # 构造新的文件名
    new_filename = str(i + 1) + '.png'

    # 构造新的文件路径
    new_filepath = os.path.join(folder_path, new_filename)

    # 重命名文件
    os.rename(image_path, new_filepath)

print("图片重命名完成！")
