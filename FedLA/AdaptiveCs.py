import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Env import Mean, Mean_TSNE, Mean_Delta
from utils.switcher import switch_n_avg, str2ndarray, switch_n_vector_avg, tsne_2dims


def plot_delta_trend(data, out: str = None):
    """
    绘制变化趋势图函数

    参数：
    data: pandas DataFrame，包含数据的DataFrame
    column_name: str，要绘制的列名

    返回值：
    无返回值，直接显示图表
    """
    data = switch_n_avg(data)
    column_data = data[Mean]
    plt.plot(column_data)
    plt.title(f"Delta——{Mean}")
    if out:
        plt.savefig(out)
    plt.show()


def plot_vector_trend(data, out: str = None):
    """
    绘制变化趋势图函数

    参数：
    data: pandas DataFrame，包含数据的DataFrame
    column_name: str，要绘制的列名

    返回值：
    无返回值，直接显示图表
    """
    data = switch_n_avg(data)
    column_data = data[Mean]
    # 绘制变化趋势图
    plt.plot(column_data)
    plt.title(f"Vector——{Mean}")
    if out:
        plt.savefig(out)
    plt.show()


def matrixs_heatmap(mean_matrices: np.ndarray, out: str = None):
    # 绘制所有 10x10 矩阵的热图
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # 调整子图布局和大小
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(mean_matrices):
            sns.heatmap(mean_matrices[i], ax=ax, cmap='viridis')
            ax.set_title(f'Sample {i + 1}')
        else:
            ax.axis('off')  # 对于空的子图，关闭坐标轴

    plt.tight_layout()

    if out:
        plt.savefig(out)
    plt.show()


def plot_if_trend(df, out: str = None):
    df = switch_n_vector_avg(df)
    avg_ifs = np.stack(df['Mean'].values)
    tsne_result = tsne_2dims(avg_ifs)
    df[Mean_TSNE] = list(map(tuple, tsne_result))

    # 提取二维坐标
    x = df[Mean_TSNE].apply(lambda coord: coord[0])  # X坐标
    y = df[Mean_TSNE].apply(lambda coord: coord[1])  # Y坐标

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.title('t-SNE Result')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.plot(x, y, marker='o')

    if out:
        plt.savefig(out)
    plt.show()

    mean_matrices = np.array([np.reshape(arr, (10, 10)) for arr in df[Mean]])

    matrixs_heatmap(mean_matrices)

    diff_matrices = np.array([mean_matrices[i + 1] - mean_matrices[i] for i in range(len(mean_matrices) - 1)])

    matrixs_heatmap(diff_matrices)


def plot_delta_heatmap(df, out):
    df = switch_n_vector_avg(df)
    mean_matrices = np.array([np.reshape(arr, (10, 10)) for arr in df[Mean]])
    matrixs_heatmap(mean_matrices)
