import os

from PIL import Image, ImageDraw


def merge_bigpng():
    """
    将下载区域瓦片合并为带有网格的大图
    :return:
    """
    # 瓦片存储目录
    workPath = r'cache\18'
    dirs = os.listdir(workPath)

    # 列号 高度
    yMin = int(dirs[0])
    yMax = int(dirs[-1])

    # 行号 宽度
    files = os.listdir(os.path.join(workPath, dirs[0]))
    xMin = int(files[0].split('.')[0])  # 列文件夹中的第一张图片名字
    xMax = int(files[-1].split('.')[0])  # 列文件夹中的最后一张图片名字

    # 瓦片的尺寸
    size = 256
    # 新建一个指定宽高的空白影像 宽高指拼接后的大图的宽高
    resultImage = Image.new('RGB', (((xMax - xMin) + 1) * size, ((yMax - yMin) + 1) * size))

    def get_image_filepath(file_path, x, y):
        return file_path + "/" + str(y) + '/' + str(x) + '.png'

    for y in range(yMin, yMax + 1):
        for x in range(xMin, xMax + 1):
            fileName = get_image_filepath(workPath, x, y)
            fromImage = Image.open(fileName)
            fromImage = fromImage.convert('RGB')
            draw = ImageDraw.Draw(fromImage)

            # # 添加信息
            # fontSize = 18  # 字号大小
            # setFont = ImageFont.truetype('C:/windows/Fonts/arial.ttf', fontSize)  # 本地电脑字体库
            # fillColor = "#ffd111"  # 字体颜色
            # text1 = 'dir:' + str(y)
            # text2 = 'file:' + str(x)
            # draw.text((1, 1), text1, font=setFont, fill=fillColor)
            # draw.text((1, 1 + fontSize), text2, font=setFont, fill=fillColor)
            #
            # # 给瓦片添加格网边框
            # draw.rectangle(((0.0, 0.0), (float(size - 1), float(size - 1))), fill=None, outline='red', width=1)

            # 将瓦片粘贴在大图
            resultImage.paste(fromImage, ((x - xMin) * size, (y - yMin) * size))

    # 完整图像存储路径
    resultImage.save(r'cache\1\1.png')
    resultImage.close()


def merge_twopng():
    """
    读取两个文件夹内的png拼接成一个512*256的png
    :return:
    """
    # 第一个路径
    path1 = r'D:\job\code_projects\wayback-download-service\cache\luzhaixian-2019-2'
    # 第二个路径
    path2 = r'D:\job\code_projects\wayback-download-service\cache\luzhaixian-2023-2'

    save_dir = r'D:\job\code_projects\wayback-download-service\cache\luzhaixian\luzhaixian2-2019-2023'

    # 遍历第一个路径下的所有文件夹和文件
    for root, dirs, files in os.walk(path1):
        for file in files:
            # 筛选出256大小的png图片
            if file.endswith('.png'):
                # 获取对应的第二个路径下的文件名
                filename2 = os.path.join(path2, os.path.relpath(root, path1), file)
                # 判断是否存在相同的文件
                if os.path.exists(filename2):
                    # 读取第一张图片
                    img1 = Image.open(os.path.join(root, file))
                    # 读取第二张图片
                    img2 = Image.open(filename2)
                    # 创建一个新的512*256大小的空白图片
                    new_img = Image.new('RGB', (512, 256))
                    # 将两张图片合并到新图片中
                    new_img.paste(img1, (0, 0))
                    new_img.paste(img2, (256, 0))
                    # 保存图片，以第一个路径下的文件夹名加文件名作为文件名
                    save_path = os.path.join(save_dir, os.path.relpath(root, path1),
                                             file.replace('.png', '_merged.png'))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    new_img.save(save_path)
                    print(f'Saved image {save_path}')


if __name__ == '__main__':
    merge_bigpng()
    # merge_twopng()
