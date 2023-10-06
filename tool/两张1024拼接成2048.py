from PIL import Image
import os

# 保存拼接后图像的输出目录
output_dir = "C:\\Users\DELL\Desktop\sj\\9\\"

# 第一个文件夹中的图片所处路径（需要拼接成左半部分）
input_dir_1 = "C:\\Users\DELL\Desktop\sj\\2\\"

# 第二个文件夹中的图片所处路径（需要拼接成右半部分）
input_dir_2 = "C:\\Users\DELL\Desktop\sj\\5\\"

def get_matched_file_paths():
    # 获取两个文件夹中所有相同文件名的图片，并返回它们的绝对路径
    file_names_1 = set(os.path.splitext(filename)[0] for filename in os.listdir(input_dir_1) if filename.endswith('.png'))
    file_names_2 = set(os.path.splitext(filename)[0] for filename in os.listdir(input_dir_2) if filename.endswith('.png'))

    matched_file_names = list(file_names_1 & file_names_2)
    matched_file_paths = []
    for file_name in matched_file_names:
        file_path_1 = os.path.join(input_dir_1, file_name + ".png")
        file_path_2 = os.path.join(input_dir_2, file_name + ".png")
        matched_file_paths.append((file_path_1, file_path_2))

    return matched_file_paths


def combine_images():
    # 获取所有匹配的图像，遍历所有匹配的图像进行拼接，并将结果保存到输出目录中
    matched_file_paths = get_matched_file_paths()
    for i, (file_path_1, file_path_2) in enumerate(matched_file_paths):
        # 打开两张图片
        img1 = Image.open(file_path_1).convert("RGBA")
        img2 = Image.open(file_path_2).convert("RGBA")

        # 确保两张图片的大小一致（1024x1024）
        img1 = img1.resize((1024, 1024))
        img2 = img2.resize((1024, 1024))

        # 创建新的空白图像，大小为2048x1024
        combined_image = Image.new('RGB', (2048, 1024), (0, 0, 0))

        # 将两张图片分别复制到新图像中的左、右两部分
        combined_image.paste(img1, (0, 0))
        combined_image.paste(img2, (1024, 0))

        # 创建输出文件名并将结果保存到输出目录中
        output_file_name = os.path.basename(file_path_1)
        output_file_path = os.path.join(output_dir, output_file_name)
        combined_image.save(output_file_path, format='PNG')


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    combine_images()
