from matplotlib import pyplot as plt

from Env import Mean
from utils.switcher import switch_n_avg


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
