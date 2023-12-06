import json
from enum import unique, Enum

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from Env import Mean


@unique
class DataType(Enum):
    Number = 1
    Vector = 2
    Matrix = 3
    Tensor = 4


def str2number(src: str):
    return float(src)


def str2ndarray(src: str):
    data_string = src.replace('\n', '').replace('[', '').replace(']', '').replace(',', ' ').replace('  ', ' ')
    ndarray_data = np.fromstring(data_string, dtype=float, sep=' ')
    return ndarray_data


def get_depth(obj):
    if isinstance(obj, list) and len(obj) > 0:
        return 1 + get_depth(obj[0])
    return 0


def identify_string_content(s):
    try:
        # 尝试解析为数字
        float(s)
        return DataType.Number
    except ValueError:
        try:
            # 尝试解析为JSON
            parsed_json = json.loads(s)
            depth = get_depth(parsed_json)
            if depth == 1:
                return DataType.Vector
            elif depth == 2:
                return DataType.Matrix
            elif depth > 2:
                return DataType.Tensor
        except (json.JSONDecodeError, TypeError):
            pass
    return "neither"


def switch_n_avg(data: pd.DataFrame):
    """
    计算所有相同列的数据的平均值，并存储在第最后一列

    参数：
    data: pandas DataFrame，包含数据的DataFrame

    返回值：
    pandas DataFrame，包含计算结果的DataFrame
    """
    column_names = data.columns.tolist()
    column_names.remove('Unnamed: 0')

    # 将 inf 和 -inf 替换为 NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 删除含有 NaN 的行
    data.dropna(inplace=True)

    # 重置索引
    data.reset_index(drop=True, inplace=True)

    # string转换为ndarray
    for col in column_names:
        func = str2number if identify_string_content(data[col][0]) == DataType.Number else str2ndarray
        column_data = data[col].apply(lambda x: func(x))
        data[col] = column_data

    data[Mean] = data.drop('Unnamed: 0', axis=1).mean(axis=1)
    return data


def switch_n_vector_avg(df: pd.DataFrame):
    column_names = df.columns.tolist()
    column_names.remove('Unnamed: 0')

    # string转换为ndarray
    for col in column_names:
        func = str2number if identify_string_content(df[col][0]) == DataType.Number else str2ndarray
        column_data = df[col].apply(lambda x: func(x))
        df[col] = column_data

    # 删除 'Unnamed: 0' 列
    df = df.drop(columns=['Unnamed: 0'])

    # 计算每行的平均值
    df[Mean] = df.apply(lambda row: np.mean(np.array(
        [row[data_col] for data_col in column_names]), axis=0), axis=1)
    return df


def tsne_2dims(data: np.ndarray, perplexity: int = 5) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=23)
    transformed_data = tsne.fit_transform(data)
    return transformed_data


def positional_encoding(positions, d_model):
    """
    计算位置编码。

    :param positions: 序列长度。
    :param d_model: 模型的维度，位置编码的维度。
    :return: 位置编码矩阵，形状为 [positions, d_model]。
    """
    # 初始化位置编码矩阵
    pos_enc = np.zeros((positions, d_model))

    # 计算位置编码
    for pos in range(positions):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return pos_enc


def cal_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算向量的欧几里得范数
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

    return cosine_similarity
