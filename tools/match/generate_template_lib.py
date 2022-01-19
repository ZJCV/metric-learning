# -*- coding: utf-8 -*-

"""
@date: 2022/1/18 下午3:33
@file: generate_templte_lib.py
@author: zj
@description: 对训练数据进行采集，创建模板特征库
"""

import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, default=None, help='csv path')
    parser.add_argument('pkl_path', type=str, default=None, help='pkl path')

    args = parser.parse_args()
    print(args)
    return args


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
    args = parse_args()
    csv_path = args.csv_path
    pkl_path = args.pkl_path

    if not os.path.exists(csv_path):
        raise ValueError(f'{csv_path} does not exists')
    if os.path.exists(pkl_path):
        raise ValueError(f'{pkl_path} has existed')

    data_array = load_data(csv_path)

    match_dict = process(data_array)
    with open(pkl_path, 'wb') as f:
        pickle.dump(match_dict, f)
