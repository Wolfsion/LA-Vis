import random

import numpy as np
import pandas as pd

from FedLA.AdaptiveCs import plot_delta_trend, plot_vector_trend
from FedLA.Matrix import plot_dis_trend, calculate_js_divergence, plot_matrix_trend
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


def matrix_dist_csv():
    seq2csv(info_matrix, outi)
    csv = pd.read_csv(outi)
    ret = calculate_js_divergence(csv)
    plot_dis_trend(ret, ['3'], outid)


def selection():
    seqs = [select8]
    seqs2csv(seqs, s_outs)
    data = pd.read_csv(s_outs)
    data = to_vis_selection(data)
    plot_cnt(data, outse)


def matrix_single():
    seq2csv(info_matrix, outi)
    csv = pd.read_csv(outi)
    plot_matrix_trend(csv, outif)


def adaptive_cs():
    # ratios = [delta_ratio1, delta_ratio2, delta_ratio3, delta_ratio4, delta_ratio5]
    # seqs2csv(ratios, delta_ratio_csv)
    # csv = pd.read_csv(delta_ratio_csv)
    # plot_delta_trend(csv, delta_ratio_img)

    js_dists = [js_dis1, js_dis2, js_dis3, js_dis4, js_dis5]
    seqs2csv(js_dists, js_dis_csv)
    csv = pd.read_csv(js_dis_csv)
    plot_vector_trend(csv, js_dis_img)


def tmp():
    pass


if __name__ == '__main__':
    # matrix_dist_csv()
    # matrix_dist_csvs()
    # selection()
    # matrix_single()
    # adaptive_cs()
    a = np.float(1)
