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
    parser.add_argument('cls_path', type=str, default=None, help='cls path')
    parser.add_argument('pkl_path', type=str, default=None, help='pkl path')

    args = parser.parse_args()
    print(args)
    return args


def load_cls(cls_path):
    assert os.path.isfile(cls_path), cls_path

    data_array = np.loadtxt(cls_path, delimiter=' ', dtype=str)

    cls_label_dict = dict()
    for label, cls in data_array:
        assert cls not in cls_label_dict.keys()
        # 对于CUB而言，label从1开始，所以需要相应的减去1
        cls_label_dict[cls] = int(label) - 1

    return cls_label_dict


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=str)
    return data_array


def process(data_array, cls_label_dict):
    match_dict = dict()

    # 遍历所有训练数据，将特征向量分别存入对应的类别中
    for item in tqdm(data_array):
        img_path = item[0]
        cls_name = os.path.split(os.path.split(img_path)[0])[1]

        pred_feats = np.array(item[1:], dtype=float)

        label = str(cls_label_dict[cls_name])
        if label not in match_dict.keys():
            match_dict[label] = list()
        match_dict[label].append(pred_feats)

    return match_dict


def save_to_pkl(match_dict, pkl_path):
    assert not os.path.isfile(pkl_path), pkl_path

    with open(pkl_path, 'wb') as f:
        pickle.dump(match_dict, f)


if __name__ == '__main__':
    args = parse_args()
    csv_path = args.csv_path
    cls_path = args.cls_path
    pkl_path = args.pkl_path

    if not os.path.exists(csv_path):
        raise ValueError(f'{csv_path} does not exists')
    if not os.path.exists(cls_path):
        raise ValueError(f'{cls_path} does not exists')
    if os.path.exists(pkl_path):
        raise ValueError(f'{pkl_path} has existed')

    cls_label_dict = load_cls(cls_path)
    data_array = load_data(csv_path)

    print('process ...')
    match_dict = process(data_array, cls_label_dict)
    print(f'save to {pkl_path}')
    save_to_pkl(match_dict, pkl_path)
