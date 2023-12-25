import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon

from Env import Mean, Mean_TSNE, Mean_Delta, Mean_Norm, Mean1
from utils.switcher import switch_n_avg, str2ndarray, switch_n_vector_avg, tsne_2dims, apply_scalar_positional_encoding, \
    remove_top_k_elements, kl_divergence, euclidean_distance


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
    avg_ifs = np.stack(df[Mean].values)
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
    #
    # matrixs_heatmap(mean_matrices)
    #
    # diff_matrices = np.array([mean_matrices[i + 1] - mean_matrices[i] for i in range(len(mean_matrices) - 1)])
    #
    # matrixs_heatmap(diff_matrices)

    js_divergences = []

    # 随机选择6个矩阵的索引
    selected_indices = random.sample(range(len(df)), 6)
    selected_indices = sorted(selected_indices)

    # range(len(df) - 1)
    for i in selected_indices:
        matrix1 = mean_matrices[i]
        matrix2 = mean_matrices[i + 1]

        # method1: 多元素
        js_div_row = []
        for j in range(len(matrix1)):
            # 完整元素
            row1 = matrix1[j]
            row2 = matrix2[j]

            # # 去除对角线元素
            # row1 = np.delete(row1, j)
            # row2 = np.delete(row2, j)

            #
            row1 = remove_top_k_elements(row1, 1)
            row2 = remove_top_k_elements(row2, 1)

            # 计算 JS 散度
            js_div = jensenshannon(row1, row2)
            # js_div = euclidean_distance(row1, row2)
            # stat, js_div = wilcoxon(row1, row2)
            # js_div = kl_divergence(row1, row2)

            js_div_row.append(js_div)

        js_divergences.append(js_div_row)

        # # # method2: 只计算对角线元素
        # diagonal1 = np.diag(matrix1)
        # diagonal2 = np.diag(matrix2)
        #
        # # 计算对角线元素的 JS 散度
        # js_div_diagonal = jensenshannon(diagonal1, diagonal2)
        # js_divergences.append(js_div_diagonal)

    # 将 JS 散度结果转换为 DataFrame
    js_df = pd.DataFrame(js_divergences)

    # # 按行取平均
    # js_df[0] = js_df.mean(axis=1)
    # js_df = js_df[[0]]  # 保留列 '0'，删除其他所有列

    # 绘制每行 JS 散度的变化趋势图
    plt.figure(figsize=(12, 8))
    for row in range(js_df.shape[1]):
        plt.plot(js_df.index, js_df[row], marker='o', label=f'Row {row + 1}')

    plt.title('Jensen-Shannon Divergence Trend for Each Row')
    plt.xlabel('Matrix Pair Index')
    plt.ylabel('JS Divergence')
    plt.legend()
    plt.grid(True)
    plt.show()

    df[Mean1] = df[Mean].apply(lambda xnd: xnd[:10])
    # 设置绘图布局，行数由 DataFrame 的长度决定
    n_rows = len(df)
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))

    # 为每个 1x10 ndarray 绘制分布曲线
    for i in range(n_rows):
        axs[i].plot(df[Mean1].iloc[i])
        axs[i].set_title(f'Distribution Curve for Element {i + 1}')
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')

    plt.tight_layout()
    plt.show()


def plot_delta_heatmap(df, out):
    df = switch_n_vector_avg(df)
    mean_matrices = np.array([np.reshape(arr, (10, 10)) for arr in df[Mean]])
    matrixs_heatmap(mean_matrices, out)


def single_line_heatmap(df, out):
    df = switch_n_vector_avg(df)
    selected_indices = random.sample(range(len(df)), 6)
    selected_indices = sorted(selected_indices)
    mean_matrices = np.array([np.reshape(arr, (10, 10)) for arr in df[Mean]])

    # 设置绘图布局
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 绘制每个矩阵第 10 行的热图
    for i, ax in zip(selected_indices, axs.flatten()):
        ax.imshow(mean_matrices[i][9, np.newaxis], aspect='auto', cmap='viridis')  # 第 10 行的索引是 9
        ax.set_title(f'Matrix {i + 1}, Row 10')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row 10 Value')
    plt.tight_layout()
    plt.show()

    row_10_arrays = [mean_matrices[i:i + 20][9] for i in selected_indices]

    # 计算相邻矩阵第 10 行向量之间的 JS 散度
    js_divergences = [jensenshannon(row_10_arrays[i], row_10_arrays[i + 1]) for i in range(len(row_10_arrays) - 1)]

    # 绘制 JS 散度的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(js_divergences)), js_divergences, marker='o')
    plt.title('Jensen-Shannon Divergence Between Adjacent Matrices (Row 10)')
    plt.xlabel('Matrix Pair Index')
    plt.ylabel('JS Divergence')
    plt.grid(True)
    plt.show()


def plot_acc_trend(df, out):
    df = switch_n_avg(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Unnamed: 0'], df['Mean'], marker='o')
    plt.title('Trend of Mean Values Over Unnamed: 0')
    plt.xlabel('Index')
    plt.ylabel('Mean Value')
    plt.grid(True)

    if out:
        plt.savefig(out)
    plt.show()
