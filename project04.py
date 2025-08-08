import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import time


def extract_txt3(csv_file_path):
    """
    提取location列（txt3）
    """
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


def locations_to_latlon(locations):
    """
    将地点列表转换为经纬度
    """
    geolocator = Nominatim(user_agent="geoapiExercises")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    results = []

    print(f"开始处理 {len(locations)} 个地点...")

    for i, loc in enumerate(locations, 1):
        try:
            print(f"正在处理第 {i}/{len(locations)} 个地点: {loc}")
            location = geocode(loc)
            if location:
                lat, lon = location.latitude, location.longitude
                print(f"  ✓ {loc} -> ({lat:.6f}, {lon:.6f})")
                results.append({'location': loc, 'lat': lat, 'lon': lon})
            else:
                print(f"  ✗ {loc} -> 未找到经纬度")
        except Exception as e:
            print(f"  ✗ {loc} -> 解析出错: {e}")

        time.sleep(1.5)

    return pd.DataFrame(results)


def perform_dbscan_clustering(df):
    """
    对经纬度数据进行DBSCAN聚类
    """
    if len(df) == 0:
        print("没有有效的经纬度数据进行聚类")
        return None, None

    coords = df[['lat', 'lon']].values

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
    """
    获取簇的中心点
    """
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def visualize_clusters(df, clusters, cluster_labels):
    """
    可视化聚类结果
    """
    if clusters is None:
        print("没有聚类结果可可视化")
        return

    # 获取代表点
    centermost_points = clusters.map(get_centermost_point)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({'lon': lons, 'lat': lats})

    # 创建图形
    fig, ax = plt.subplots(figsize=[12, 8])

    # 绘制原始点（黑色小点）
    df_scatter = ax.scatter(df['lon'], df['lat'], c='k', alpha=0.6, s=20, label='原始数据点')

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
        cluster_points = df[mask]
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


def analyze_clusters(df, cluster_labels):
    """
    分析聚类结果
    """
    if cluster_labels is None:
        return

    unique_labels = set(cluster_labels)
    num_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)

    print(f"\n=== 聚类分析结果 ===")
    print(f"总数据点数量: {len(df)}")
    print(f"聚类数量: {num_clusters}")
    print(f"噪声点数量: {num_noise}")
    print(f"聚类点数量: {len(df) - num_noise}")

    # 分析每个聚类的大小
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            size = list(cluster_labels).count(label)
            cluster_sizes.append(size)
            print(f"聚类 {label}: {size} 个点")

    if cluster_sizes:
        print(f"\n聚类大小统计:")
        print(f"最大聚类大小: {max(cluster_sizes)}")
        print(f"最小聚类大小: {min(cluster_sizes)}")
        print(f"平均聚类大小: {np.mean(cluster_sizes):.1f}")


if __name__ == "__main__":
    csv_file = 'data.csv'  # 替换为你的CSV文件名

    # 检查文件是否存在
    import os

    if not os.path.exists(csv_file):
        print(f"错误：文件 '{csv_file}' 不存在")
        print("请确保CSV文件在当前目录下，或修改代码中的文件名")
    else:
        # 步骤1：提取地点数据
        txt3 = extract_txt3(csv_file)
        if txt3:
            print(f"\n成功提取到 {len(txt3)} 个地点:")
            for i, loc in enumerate(txt3, 1):
                print(f"{i}. {loc}")

            # 步骤2：转换为经纬度
            print(f"\n开始转换为经纬度...")
            df_latlon = locations_to_latlon(txt3)

            if len(df_latlon) > 0:
                print(f"\n成功转换 {len(df_latlon)} 个地点的经纬度")

                # 步骤3：执行DBSCAN聚类
                print(f"\n开始执行DBSCAN聚类...")
                clusters, cluster_labels = perform_dbscan_clustering(df_latlon)

                if clusters is not None:
                    # 步骤4：可视化聚类结果
                    print(f"\n生成聚类可视化图...")
                    rep_points = visualize_clusters(df_latlon, clusters, cluster_labels)

                    # 步骤5：分析聚类结果
                    analyze_clusters(df_latlon, cluster_labels)

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