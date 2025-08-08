import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np


def extract_columns_to_variables(csv_file_path):
    """
    从CSV文件中提取指定列的数据，并将每列数据保存到变量中。
    txt1为id，其他列依次往后移动。

    Args:
        csv_file_path (str): CSV文件的路径

    Returns:
        dict: 包含所有提取数据的字典，键为txt1, txt2, txt3等
    """
    try:
        # 尝试不同的编码格式读取CSV文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)  # 修复：使用传入的参数
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print("错误：无法使用任何编码格式读取文件")
            return None

        # 根据图片中的分类定义要提取的列，txt1为id，其他依次往后
        columns_to_extract = [
            "id",  # txt1
            "name",  # txt2
            "location",  # txt3
            "linkedin_url",  # txt4
            "gender",  # txt5
            "relationship",  # txt6
            "workplace",  # txt7
            "birthdate"  # txt8
        ]

        # 检查所有目标列是否存在于DataFrame中
        missing_columns = [col for col in columns_to_extract if col not in df.columns]
        if missing_columns:
            print(f"警告：CSV文件中缺少以下列：{missing_columns}")
            columns_to_extract = [col for col in columns_to_extract if col in df.columns]

        # 创建字典来存储提取的数据
        extracted_data = {}

        # 遍历每一列，提取数据并保存到对应的变量中
        for i, col_name in enumerate(columns_to_extract):
            variable_name = f"txt{i + 1}"

            if col_name == "workplace":  # 特殊处理txt7
                # 直接提取为可迭代对象（列表）
                column_data = df[col_name].dropna().astype(str).tolist()
                # 过滤空字符串
                column_data = [item.strip() for item in column_data if item.strip()]
            else:
                # 其他列保持原来的字符串格式
                column_data = df[col_name].dropna().astype(str).str.cat(sep='\n')

            # 保存到字典中
            extracted_data[variable_name] = column_data
            print(f"列 '{col_name}' 的数据已保存到变量 '{variable_name}'")

        print(f"\n成功提取了 {len(columns_to_extract)} 列数据到变量中。")

        return extracted_data

    except FileNotFoundError:
        print(f"错误：文件 '{csv_file_path}' 未找到")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def analyze_relationship_frequency(txt6_data):
    """
    统计txt6中不同情感关系状况的频率并生成柱状图

    Args:
        txt6_data: txt6变量（relationship列数据）
    """
    # 如果txt6是字符串，转换为列表
    if isinstance(txt6_data, str):
        relationship_list = txt6_data.split('\n')
        relationship_list = [item.strip() for item in relationship_list if item.strip()]
    else:
        relationship_list = txt6_data

    # 统计频率
    frequency_counter = Counter(relationship_list)

    # 转换为DataFrame并按频率倒序排序
    df_freq = pd.DataFrame(list(frequency_counter.items()), columns=['Relationship', 'Count'])
    df_freq = df_freq.sort_values('Count', ascending=False)

    # 计算百分比
    total_count = len(relationship_list)
    df_freq['Percentage'] = (df_freq['Count'] / total_count * 100).round(2)

    # 打印统计结果
    print("=== 情感关系状况统计 ===")
    print(f"总数据条数: {total_count}")
    print(f"不同关系类型数: {len(frequency_counter)}")
    print("\n频率统计:")
    print("-" * 50)
    print(f"{'关系状况':<20} {'数量':<8} {'百分比':<8}")
    print("-" * 50)

    for index, row in df_freq.iterrows():
        relationship = row['Relationship']
        count = row['Count']
        percentage = row['Percentage']
        print(f"{relationship:<20} {count:<8} {percentage:<8.2f}%")

    print("-" * 50)

    return df_freq


def create_relationship_chart(txt6_data):
    """
    创建情感关系状况的柱状图
    """
    # 如果txt6是字符串，转换为列表
    if isinstance(txt6_data, str):
        relationship_list = txt6_data.split('\n')
        relationship_list = [item.strip() for item in relationship_list if item.strip()]
    else:
        relationship_list = txt6_data

    # 统计频率
    frequency_counter = Counter(relationship_list)
    df_freq = pd.DataFrame(list(frequency_counter.items()), columns=['Relationship', 'Count'])
    df_freq = df_freq.sort_values('Count', ascending=False)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 第一个子图：普通柱状图
    bars1 = ax1.bar(range(len(df_freq)), df_freq['Count'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                    edgecolor='navy', alpha=0.8)

    ax1.set_title('情感关系状况频率统计', fontsize=14, fontweight='bold')
    ax1.set_xlabel('关系状况', fontsize=12)
    ax1.set_ylabel('人数', fontsize=12)
    ax1.set_xticks(range(len(df_freq)))
    ax1.set_xticklabels(df_freq['Relationship'], rotation=45, ha='right')

    # 在柱子上添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    ax1.grid(True, alpha=0.3, axis='y')

    # 第二个子图：水平柱状图
    bars2 = ax2.barh(range(len(df_freq)), df_freq['Count'],
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                     edgecolor='navy', alpha=0.8)

    ax2.set_title('情感关系状况频率统计（水平）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('人数', fontsize=12)
    ax2.set_ylabel('关系状况', fontsize=12)
    ax2.set_yticks(range(len(df_freq)))
    ax2.set_yticklabels(df_freq['Relationship'])

    # 在水平柱子上添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{int(width)}', ha='left', va='center', fontweight='bold')

    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


def create_seaborn_chart(txt6_data):
    """
    使用seaborn创建更美观的图表
    """
    # 如果txt6是字符串，转换为列表
    if isinstance(txt6_data, str):
        relationship_list = txt6_data.split('\n')
        relationship_list = [item.strip() for item in relationship_list if item.strip()]
    else:
        relationship_list = txt6_data

    # 统计频率
    frequency_counter = Counter(relationship_list)
    df_freq = pd.DataFrame(list(frequency_counter.items()), columns=['Relationship', 'Count'])
    df_freq = df_freq.sort_values('Count', ascending=False)

    # 创建seaborn图表
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # 创建柱状图
    ax = sns.barplot(data=df_freq, x='Relationship', y='Count',
                     palette='husl', alpha=0.8)

    # 设置标题和标签
    plt.title('情感关系状况频率统计 (Seaborn)', fontsize=16, fontweight='bold')
    plt.xlabel('关系状况', fontsize=12)
    plt.ylabel('人数', fontsize=12)

    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')

    # 在柱子上添加数值
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


# 主程序执行
if __name__ == "__main__":
    # 使用示例
    csv_file = 'data.csv'  # 请替换为您的CSV文件路径
    data_variables = extract_columns_to_variables(csv_file)

    if data_variables:
        # 现在可以通过字典访问各个变量的数据
        txt1 = data_variables['txt1']  # id列数据（字符串）
        txt2 = data_variables['txt2']  # name列数据（字符串）
        txt3 = data_variables['txt3']  # location列数据（字符串）
        txt4 = data_variables['txt4']  # linkedin_url列数据（字符串）
        txt5 = data_variables['txt5']  # gender列数据（字符串）
        txt6 = data_variables['txt6']  # relationship列数据（字符串）
        txt7 = data_variables['txt7']  # workplace列数据（可迭代对象/列表）
        txt8 = data_variables['txt8']  # birthdate列数据（字符串）

        # 分析txt6数据
        relationship_stats = analyze_relationship_frequency(txt6)

        # 创建普通柱状图
        create_relationship_chart(txt6)

        # 创建seaborn图表（可选）
        create_seaborn_chart(txt6)

        # 保存统计结果到CSV文件（可选）
        relationship_stats.to_csv('relationship_frequency.csv', index=False)
        print("\n统计结果已保存到 'relationship_frequency.csv'")

        # 显示前3个最常见的关系状况
        print("\n前3个最常见的情感关系状况:")
        print(relationship_stats.head(3))

    else:
        print("请检查CSV文件路径是否正确")


# 如果您想要更详细的分析
def detailed_relationship_analysis(txt6_data):
    """
    更详细的情感关系分析
    """
    if isinstance(txt6_data, str):
        relationship_list = txt6_data.split('\n')
        relationship_list = [item.strip() for item in relationship_list if item.strip()]
    else:
        relationship_list = txt6_data

    frequency_counter = Counter(relationship_list)
    df_freq = pd.DataFrame(list(frequency_counter.items()), columns=['Relationship', 'Count'])
    df_freq = df_freq.sort_values('Count', ascending=False)

    print("\n=== 详细分析 ===")
    print(f"空值数量: {txt6_data.count('') if isinstance(txt6_data, str) else 0}")
    print(f"唯一关系类型数量: {len(frequency_counter)}")
    print(f"最常见的关系状况: {df_freq.iloc[0]['Relationship']} ({df_freq.iloc[0]['Count']}人)")
    print(f"最少见的关系状况: {df_freq.iloc[-1]['Relationship']} ({df_freq.iloc[-1]['Count']}人)")

    # 统计只出现一次的关系状况
    single_occurrence = df_freq[df_freq['Count'] == 1]
    print(f"只出现一次的关系状况数量: {len(single_occurrence)}")

    # 计算多样性指数
    total = len(relationship_list)
    diversity = len(frequency_counter) / total if total > 0 else 0
    print(f"关系状况多样性指数: {diversity:.3f}")

    return df_freq

# 运行详细分析（可选）
# detailed_relationship_analysis(txt6)