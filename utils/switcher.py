import numpy as np


def str2ndarray(src: str):
    data_string = src.replace('\n', '').replace('[', '').replace(']', '').replace(',', ' ').replace('  ', ' ')
    ndarray_data = np.fromstring(data_string, dtype=float, sep=' ')
    return ndarray_data
