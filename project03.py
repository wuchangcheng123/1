import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_columns_to_variables(csv_file_path):
    """
    从CSV文件中提取指定列的数据，并将每列数据保存到变量中。
    txt8为birthdate列。
    """
    try:
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
        if 'birthdate' not in df.columns:
            print("错误：CSV文件中没有birthdate列")
            return None
        # 只提取birthdate列
        txt8 = df['birthdate'].dropna().astype(str).tolist()
        txt8 = [item.strip() for item in txt8 if item.strip()]
        return txt8
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def calculate_age_from_birthdate(txt8_data):
    """
    从birthdate数据计算年龄，返回年龄列表
    """
    ages = []
    current_year = datetime.now().year
    for birthdate in txt8_data:
        try:
            date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
            age = None
            for fmt in date_formats:
                try:
                    birth_date = datetime.strptime(birthdate, fmt)
                    age = current_year - birth_date.year
                    if (datetime.now().month, datetime.now().day) < (birth_date.month, birth_date.day):
                        age -= 1
                    break
                except ValueError:
                    continue
            if age is not None and 0 <= age <= 120:
                ages.append(age)
        except Exception:
            continue
    return ages

def plot_age_histogram(ages):
    """
    绘制年龄分布直方图
    """
    if not ages:
        print("没有有效的年龄数据")
        return
    plt.figure(figsize=(10,6))
    plt.hist(ages, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('年龄分布直方图', fontsize=16)
    plt.xlabel('年龄')
    plt.ylabel('人数')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    csv_file = 'data.csv'  # 替换为你的CSV文件名
    txt8 = extract_columns_to_variables(csv_file)
    if txt8:
        ages = calculate_age_from_birthdate(txt8)
        plot_age_histogram(ages)
    else:
        print("未能正确提取birthdate数据")