<!--
 * @Author: wupeiwen <javapeiwen2010@gmail.com>
 * @Date: 2023-07-11 18:39:28
 * @LastEditors: wupeiwen <javapeiwen2010@gmail.com>
 * @LastEditTime: 2023-07-11 18:50:22
 * @FilePath: /python-download-tile/readme.md
 * @Description: 说明文件
-->
## 下载依赖
```
pip3 install -r requirements.txt
```

## 更改配置
在`main.py`中，更改query参数
```
    # 配置信息
    query = {
        # 图层ID
        'id': 3201,
        # 目标区域的BBOX (minX, minY, maxX, maxY)
        'bbox': '119.11,27.39,119.58,28.11',
        # 地图缩放等级，最大为18级
        'z': 17
    }
```

## 启动程序
```
python3 main.py
```