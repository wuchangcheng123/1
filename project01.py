import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
                df = pd.read_csv(csv_file_path, encoding=encoding)
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


def create_wordcloud_from_txt7(txt7_data):
    """
    从txt7数据创建词云

    Args:
        txt7_data: txt7变量（应该是列表格式）
    """
    if isinstance(txt7_data, str):
        # 如果是字符串，先转换为列表
        txt7_list = txt7_data.split('\n')
        txt7_list = [item.strip() for item in txt7_list if item.strip()]
    else:
        # 如果已经是列表，直接使用
        txt7_list = txt7_data

    # 统计频率
    frequency_counter = Counter(txt7_list)

    # 创建词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate_from_frequencies(frequency_counter)

    # 显示词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Workplace 词云图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return frequency_counter


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

    # 验证txt7是可迭代对象
    print(f"txt7 类型: {type(txt7)}")
    print(f"txt7 长度: {len(txt7)}")
    print(f"txt7 内容: {txt7}")

    # 创建词云
    frequency_dict = create_wordcloud_from_txt7(txt7)
    print(f"\n频率统计: {dict(frequency_dict)}")

else:
    print("请先运行数据提取代码获取txt7变量")


