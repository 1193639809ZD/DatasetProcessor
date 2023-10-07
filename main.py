'''
Author: wupeiwen <javapeiwen2010@gmail.com>
Date: 2023-07-11 18:05:51
LastEditors: wupeiwen <javapeiwen2010@gmail.com>
LastEditTime: 2023-07-11 18:50:15
FilePath: /python-download-tile/main.py
Description: 瓦片计算与下载
'''
import os
import aiohttp
import math
import asyncio
from datetime import datetime

# 日志文件 
log_file_name = 'cache.log'
log_stream = None

# 最大并发数
max_concurrency = 10

# 缓存
cache = {}


# 下载单张瓦片
async def download_tile(tile_url, cache_file_path):
    async with aiohttp.ClientSession() as session:
        async with session.get(tile_url) as response:
            if response.status == 200:
                with open(cache_file_path, 'wb') as f:
                    f.write(await response.read())
                cache[cache_file_path] = True
                print(f"{datetime.now().isoformat()} Tile downloaded and cached: {cache_file_path}")
            else:
                message = f"Failed to get tile: {tile_url}, {response.status}"
                print(message)
                log_stream.write(f"{datetime.now().isoformat()} {message}\n")


# 查询瓦片总数，并按照并发数控制下载
def query_tiles():
    global log_stream
    # 配置信息
    query = {
        # 图层ID
        'id': 47963,
        # 目标区域的BBOX (minX, minY, maxX, maxY)
        'bbox': '108.964882,34.487270,109.062880,34.574523',
        # 地图缩放等级，最大为18级
        'z': 17
    }

    TILE_URL = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/GoogleMapsCompatible/MapServer/tile/{query['id']}"
    bbox = list(map(float, query['bbox'].split(',')))
    z = int(query['z'])
    minX = int((bbox[0] + 180) / 360 * pow(2, z))
    minY = int(
        (1 - math.log(math.tan(bbox[3] * math.pi / 180) + 1 / math.cos(bbox[3] * math.pi / 180)) / math.pi) / 2 * pow(2,
                                                                                                                      z))
    maxX = int((bbox[2] + 180) / 360 * pow(2, z))
    maxY = int(
        (1 - math.log(math.tan(bbox[1] * math.pi / 180) + 1 / math.cos(bbox[1] * math.pi / 180)) / math.pi) / 2 * pow(2,
                                                                                                                      z))
    total_count = ((maxY - minY + 1) * (maxX - minX + 1))
    cached_count = 0
    print(f"--- Total tiles: {total_count} ---")

    if log_stream is None:
        log_stream = open(log_file_name, 'a')

    loop = asyncio.get_event_loop()
    count = 0
    tasks = []
    for y in range(minY, maxY + 1):
        for x in range(minX, maxX + 1):
            cache_file_path = os.path.join(os.getcwd(), f"cache/{z}/{y}/{x}.png")
            if cache_file_path in cache:
                continue

            # 创建目录
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

            tile_url = f"{TILE_URL}/{z}/{y}/{x}"
            task = asyncio.ensure_future(download_tile(tile_url, cache_file_path))
            tasks.append(task)
            count += 1

            if count >= max_concurrency:
                completed, _ = loop.run_until_complete(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                for completed_task in completed:
                    completed_task.result()
                tasks = []
                count -= 1

    for task in tasks:
        task.result()

    log_stream.close()

    print(f"Total tiles: {total_count}, cached tiles: {cached_count}")


# 运行
query_tiles()
