import pandas as pd

from FedLA.Matrix import plot_trend, calculate_js_divergence
from FedLA.Selection import to_vis_selection, plot_cnt, test_matrix_vis
from Paths import *
from utils.objectIO import seqs2csv


def single():
    # 读取CSV文件
    data = pd.read_csv('res/mat.csv')

    # 调用计算函数
    result = calculate_js_divergence(data)

    # 调用绘图函数，绘制列3的变化趋势图
    plot_trend(result, '3')


def csvs():
    seqs = [seq5, seq6, seq7, seq8, seq9]
    seqs2csv(seqs, outs)

    csv = pd.read_csv(outs)
    ret = calculate_js_divergence(csv)
    plot_trend(ret, ['6', '7', '8', '9', '10', '11'])


def selection():
    seqs = [select1, select2, select3, select4, select5]
    seqs2csv(seqs, s_outs)
    data = pd.read_csv(s_outs)
    data = to_vis_selection(data)
    plot_cnt(data)


if __name__ == '__main__':
    # csvs()
    # selection()
    test_matrix_vis()
