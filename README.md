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