import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt


def calculate_js_divergence(data):
    """
    计算相邻两个ndarray的JS散度，并存储在第三列

    参数：
    data: pandas DataFrame，包含数据的DataFrame

    返回值：
    pandas DataFrame，包含计算结果的DataFrame
    """
    # 获取第一列数据
    column_data = data['1']

    # 初始化第三列
    data['3'] = np.nan

    # 计算JS散度并存储在第三列
    for i in range(1, len(column_data)):
        js_divergence = jensenshannon(column_data[i - 1], column_data[i])
        data.at[i, '3'] = js_divergence

    return data


def plot_trend(data, column_name):
    """
    绘制变化趋势图函数

    参数：
    data: pandas DataFrame，包含数据的DataFrame
    column_name: str，要绘制的列名

    返回值：
    无返回值，直接显示图表
    """
    column_data = data[column_name]

    # 绘制变化趋势图
    plt.plot(column_data)

    # 添加标题和标签
    plt.title("变化趋势图")
    plt.xlabel("数据点")
    plt.ylabel("数据值")

    # 显示图表
    plt.show()

