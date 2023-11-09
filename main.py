import pandas as pd

from FedLA.Matrix import plot_dis_trend, calculate_js_divergence, plot_trend
from FedLA.Selection import to_vis_selection, plot_cnt
from Paths import *
from utils.objectIO import seqs2csv, seq2csv


def dist_single():
    # 读取CSV文件
    data = pd.read_csv('res/mat.csv')

    # 调用计算函数
    result = calculate_js_divergence(data)

    # 调用绘图函数，绘制列3的变化趋势图
    plot_dis_trend(result, '3')


def matrix_dist_csvs():
    seqs = [seq5, seq6, seq7, seq8, seq9]
    seqs2csv(seqs, outs)

    csv = pd.read_csv(outs)
    ret = calculate_js_divergence(csv)
    plot_dis_trend(ret, ['6', '7', '11'], outch)


def selection():
    seqs = [select8]
    seqs2csv(seqs, s_outs)
    data = pd.read_csv(s_outs)
    data = to_vis_selection(data)
    plot_cnt(data, outse)


def matrix_single():
    seq2csv(info_matrix, outi)
    csv = pd.read_csv(outi)
    plot_trend(csv, outif)


def tmp():
    pass


if __name__ == '__main__':
    matrix_dist_csvs()
    # selection()
    # matrix_single()
    # tmp()
