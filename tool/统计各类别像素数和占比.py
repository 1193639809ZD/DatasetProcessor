import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# 类别名称对应的像素值范围
label_ranges = {'others': (0, 0), 'forest': (1, 1), 'vegetion': (2, 2), 'water': (3, 3), 'building': (4, 4), 'bareland': (5, 5), 'grasslang': (6, 6)}
# 标签名称对应的像素数
label_counts = {label: 0 for label in label_ranges}
# 总像素数
total_count = 0

# 遍历文件夹中的所有文件
for filename in os.listdir('C:\\Users\DELL\\Desktop\\sj\\4\\'):
    # 判断文件是否是 png 图像文件
    if filename.endswith('.png'):
        # 打开图像，并获取像素数据
        with Image.open(os.path.join('C:\\Users\DELL\\Desktop\\sj\\4\\', filename)) as img:
            pixels = img.load()
            width, height = img.size
            # 统计每个标签的像素数
            for y in range(height):
                for x in range(width):
                    pixel_value = pixels[x, y]
                    for label, (start, end) in label_ranges.items():
                        if start <= pixel_value <= end:
                            label_counts[label] += 1
                            break
                    total_count += 1

# 创建 DataFrame 对象
data = {'类别': list(label_counts.keys()), '像素点个数': list(label_counts.values())}
df = pd.DataFrame(data)

# 计算每个类别的像素数占总像素的比例，并进行格式化
df['占比'] = df['像素点个数'] / total_count * 100
df['占比'] = df['占比'].map('{:.2f}%'.format)

# 将 DataFrame 保存为 Excel 文件
output_path = 'C:\\Users\DELL\\Desktop\\sj\\yueyang区域二.xlsx'
df.to_excel(output_path, index=False)

# 绘制像素类别占比的条形统计图
labels = df['类别']
counts = df['占比']

# 将字符串类型的百分比转换为浮点数
counts = [float(count.replace('%', '')) for count in counts]

# 创建保存图表的文件夹
chart_folder = os.path.dirname(output_path)
chart_path = os.path.join(chart_folder, '像素类别占比统计图.png')

plt.bar(labels, counts)

# 添加标注
for i in range(len(counts)):
    plt.text(i, counts[i], str(counts[i]), ha='center', va='bottom')

# 设置y轴刻度范围
plt.ylim(0, max(counts) * 1.1)

# 显示图表
plt.savefig(chart_path)
plt.close()

# 输出结果
print('结果已保存至：', output_path)
print('总共的像素点个数：', total_count)
print('统计图已保存至：', chart_path)
