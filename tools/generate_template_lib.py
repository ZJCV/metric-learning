# -*- coding: utf-8 -*-

"""
@date: 2022/1/18 下午3:33
@file: generate_templte_lib.py
@author: zj
@description: 对训练数据进行采集，创建模板特征库，每个类别最多采集N条数据
"""

import os
import pickle

import numpy as np
from tqdm import tqdm


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=float)

    return data_array


def process(data_array):
    label_array = np.unique(data_array[:, 0])
    match_dict = dict()
    for label in label_array:
        match_dict[int(label)] = list()

    # 遍历所有训练数据，将特征向量分别存入对应的类别中
    for item in tqdm(data_array):
        truth_label = int(item[0])
        pred_feats = item[1:]

        match_dict[truth_label].append(pred_feats)

    return match_dict


if __name__ == '__main__':
    csv_path = './outputs/train_logits.csv'
    data_array = load_data(csv_path)

    match_dict = process(data_array)
    with open('./outputs/train_template_lib.pkl', 'wb') as f:
        pickle.dump(match_dict, f)
