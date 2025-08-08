import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import requests
import time


def extract_txt3(csv_file_path):
    """提取location列（txt3）"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue
    if df is None or 'location' not in df.columns:
        print("未能正确提取location数据")
        return None
    txt3 = df['location'].dropna().astype(str).tolist()
    txt3 = [item.strip() for item in txt3 if item.strip()]
    return txt3


def amap_geocode(locations, api_key):
    """
    使用高德地图API进行地理编码
    需要先申请高德地图API密钥：https://lbs.amap.com/
    """
    results = []

    for i, loc in enumerate(locations, 1):
        try:
            print(f"正在处理第 {i}/{len(locations)} 个地点: {loc}")

            # 高德地图地理编码API
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {
                'address': loc,
                'key': api_key,
                'output': 'json'
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data['status'] == '1' and data['geocodes']:
                location = data['geocodes'][0]['location']
                lon, lat = location.split(',')
                print(f"  ✓ {loc} -> ({lat}, {lon})")
                results.append({'location': loc, 'lat': float(lat), 'lon': float(lon)})
            else:
                print(f"  ✗ {loc} -> 未找到经纬度")
                results.append({'location': loc, 'lat': None, 'lon': None})

        except Exception as e:
            print(f"  ✗ {loc} -> 解析出错: {e}")
            results.append({'location': loc, 'lat': None, 'lon': None})

        time.sleep(0.5)  # 高德API限制每秒2次请求

    return pd.DataFrame(results)


def perform_dbscan_clustering(df):
    """对经纬度数据进行DBSCAN聚类"""
    if len(df) == 0:
        print("没有有效的经纬度数据进行聚类")
        return None, None

    # 过滤掉无效的经纬度
    df_valid = df.dropna(subset=['lat', 'lon'])
    if len(df_valid) == 0:
        print("没有有效的经纬度数据")
        return None, None

    coords = df_valid[['lat', 'lon']].values

    # 地球平均半径，用于将公里转换为弧度距离
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian  # 将1.5公里转换为弧度

    # 执行DBSCAN聚类
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))

    print(f'聚类数量: {num_clusters}')

    # 创建聚类结果
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

    return clusters, cluster_labels


def get_centermost_point(cluster):
    """获取簇的中心点"""
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def visualize_clusters(df, clusters, cluster_labels):
    """可视化聚类结果"""
    if clusters is None:
        print("没有聚类结果可可视化")
        return

    # 过滤掉无效数据
    df_valid = df.dropna(subset=['lat', 'lon'])

    # 获取代表点
    centermost_points = clusters.map(get_centermost_point)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({'lon': lons, 'lat': lats})

    # 创建图形
    fig, ax = plt.subplots(figsize=[12, 8])

    # 绘制原始点（黑色小点）
    df_scatter = ax.scatter(df_valid['lon'], df_valid['lat'], c='k', alpha=0.6, s=20, label='原始数据点')

    # 绘制代表点（红色大点）
    rep_scatter = ax.scatter(rep_points['lon'], rep_points['lat'],
                             c='red', edgecolor='black', alpha=0.8, s=100, label='聚类代表点')

    # 为每个聚类绘制不同颜色的点
    unique_labels = set(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:  # 噪声点
            continue
        mask = cluster_labels == label
        cluster_points = df_valid[mask]
        ax.scatter(cluster_points['lon'], cluster_points['lat'],
                   c=[color], alpha=0.5, s=30, label=f'聚类 {label}')

    ax.set_title('地点聚类分析结果', fontsize=16, fontweight='bold')
    ax.set_xlabel('经度 (Longitude)', fontsize=12)
    ax.set_ylabel('纬度 (Latitude)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return rep_points


if __name__ == "__main__":
    csv_file = 'data.csv'

    # 检查文件是否存在
    import os

    if not os.path.exists(csv_file):
        print(f"错误：文件 '{csv_file}' 不存在")
    else:
        # 步骤1：提取地点数据
        txt3 = extract_txt3(csv_file)
        if txt3:
            print(f"\n成功提取到 {len(txt3)} 个地点")

            # 步骤2：使用高德地图API转换为经纬度
            # 需要申请高德地图API密钥：https://lbs.amap.com/
            api_key = "621fb07263742b7226ce65abc8c6a81b"  # 替换为你的高德API密钥

            if api_key == "your_amap_api_key_here":
                print("请先申请高德地图API密钥并替换代码中的api_key")
                print("申请地址：https://lbs.amap.com/")
            else:
                print(f"\n开始转换为经纬度...")
                df_latlon = amap_geocode(txt3, api_key)

                if len(df_latlon) > 0:
                    print(f"\n成功转换 {len(df_latlon)} 个地点的经纬度")

                    # 步骤3：执行DBSCAN聚类
                    print(f"\n开始执行DBSCAN聚类...")
                    clusters, cluster_labels = perform_dbscan_clustering(df_latlon)

                    if clusters is not None:
                        # 步骤4：可视化聚类结果
                        print(f"\n生成聚类可视化图...")
                        rep_points = visualize_clusters(df_latlon, clusters, cluster_labels)

                        # 保存结果
                        df_latlon.to_csv('location_latlon.csv', index=False)
                        rep_points.to_csv('cluster_representatives.csv', index=False)
                        print(f"\n结果已保存:")
                        print(f"- 完整经纬度数据: location_latlon.csv")
                        print(f"- 聚类代表点: cluster_representatives.csv")

                    else:
                        print("聚类失败")
                else:
                    print("没有成功转换的经纬度数据")
        else:
            print("未能正确提取location数据")