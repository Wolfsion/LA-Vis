import pickle

import pandas as pd

from FedLA.Matrix import plot_trend, calculate_js_divergence
from Paths import *
from objectIO import seq2csv, seqs2csv

if __name__ == '__main__':
    # 读取CSV文件
    data = pd.read_csv('res/mat.csv')

    # 调用计算函数
    result = calculate_js_divergence(data)

    # 调用绘图函数，绘制列3的变化趋势图
    plot_trend(result, '3')
