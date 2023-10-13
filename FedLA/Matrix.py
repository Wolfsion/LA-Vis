import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

from utils.switcher import str2ndarray


def calculate_js_divergence(data):
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

    # 创建新列名列表
    ori_len = len(column_names)
    tail = int(column_names[-1])
    new_column_names = []
    for i in range(ori_len + 1):
        data[str(tail + i + 1)] = np.nan
        new_column_names.append(tail + i + 1)
    new_column_names = [str(x) for x in new_column_names]

    for ind, col in enumerate(column_names):
        column_data = data[col]
        # 计算JS散度并存储在第三列
        for i in range(1, len(column_data)):
            js_divergence = jensenshannon(column_data[i - 1], column_data[i])
            data.at[i, new_column_names[ind]] = js_divergence

    # 计算平均值
    mean_value = data[new_column_names].mean(axis=1)

    # 将平均值填充到最后列
    data[new_column_names[-1]] = mean_value

    return data


def plot_trend(data, column_names):
    """
    绘制变化趋势图函数

    参数：
    data: pandas DataFrame，包含数据的DataFrame
    column_name: str，要绘制的列名

    返回值：
    无返回值，直接显示图表
    """

    for column_name in column_names:
        column_data = data[column_name]
        # 绘制变化趋势图
        plt.plot(column_data, label=column_name)

    # 添加标题和标签
    plt.title("Chart of Changing Trend")
    plt.xlabel("JS-Distance")
    plt.ylabel("Amplitude")
    plt.legend()

    # 显示图表
    plt.show()
