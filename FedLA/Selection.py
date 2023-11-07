import random
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.switcher import str2ndarray


def count_numbers(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_values, counts))


def to_vis_selection(data):
    """
    计算相邻两个ndarray的JS散度，并存储在第三列

    参数：
    data: pandas DataFrame，包含数据的DataFrame

    返回值：
    pandas DataFrame，包含计算结果的DataFrame
    """
    column_names = data.columns.tolist()
    column_names.remove('Unnamed: 0')

    # string转换为ndarray
    for col in column_names:
        column_data = data[col].apply(lambda x: str2ndarray(x))
        data[col] = column_data
    return data


def plot_cnt(data):
    """
    绘制变化趋势图函数

    参数：
    data: pandas DataFrame，包含数据的DataFrame
    column_name: str，要绘制的列名

    返回值：
    无返回值，直接显示图表
    """
    column_names = data.columns.tolist()
    column_names.remove('Unnamed: 0')

    stats_results = []

    # 遍历DataFrame中的每一列
    for column in column_names:
        summary_counts = {}
        # 统计当前列中各个数的个数
        number_counts = data[column].apply(count_numbers)

        # 将统计结果加和到汇总统计字典中
        for counts in number_counts:
            for number, count in counts.items():
                summary_counts[number] = summary_counts.get(number, 0) + count

        stats_results.append(deepcopy(summary_counts))

    # 计算平均值
    all_numbers = np.unique([number for result in stats_results for number in result.keys()])

    # 创建结果字典，并初始化为0
    result_dict = {number: 0 for number in all_numbers}

    # 统计每个数值出现的次数
    for result in stats_results:
        for number in result.keys():
            result_dict[number] += result[number]

    # 计算每个数值的平均出现个数
    mean_results = {number: count / len(stats_results) for number, count in result_dict.items()}

    # 打印平均值
    print(mean_results)

    mean_results = dict(random.sample(mean_results.items(), 10))

    # 将字典转换为 DataFrame 格式
    df = pd.DataFrame(list(mean_results.items()), columns=['X', 'Y'])

    # 使用 Seaborn 绘制条形柱状图
    sns.barplot(x='X', y='Y', data=df, label='Cnt')

    # 添加标题和标签
    plt.title("Client Selection Statics")
    plt.xlabel("Client index")
    plt.ylabel("Cnt")
    plt.legend()

    # 显示图表
    plt.show()


