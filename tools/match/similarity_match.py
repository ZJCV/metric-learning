# -*- coding: utf-8 -*-

"""
@date: 2022/1/18 下午2:30
@file: template_match.py
@author: zj
@description: 
1. 读取测试样本的特征向量列表；
2. 依次遍历：
    2.1 如果模板库不存在，则该测试样本计算错误。将测试样本加入特征库
    2.2 将测试样本和模板库进行计算，计算top1/top5结果。将测试样本加入特征库
    2.3 特征库每个类别保存的特征模板数固定，如果超出，则按先进先出策略进行替换。
3. 最后统计top1/top5准确率
"""

import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', type=str, default=None, help='test path')
    parser.add_argument('--N', type=int, default=5, help='number of each class in match lib')
    parser.add_argument('--match_path', type=str, default=None, help='match path')

    args = parser.parse_args()
    print(args)
    return args


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=float)

    return data_array


def load_match_lib(pkl_path):
    assert os.path.isfile(pkl_path), pkl_path

    with open(pkl_path, 'rb') as f:
        match_dict = pickle.load(f)

    return match_dict


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def process(data_array, match_dict, N):
    label_array = np.unique(data_array[:, 0])
    for label in label_array:
        if label not in match_dict.keys():
            match_dict[int(label)] = list()

    top1_num = 0
    top5_num = 0

    arr_len = len(data_array)
    for item in tqdm(data_array):
        truth_label = int(item[0])
        pred_feats = item[1:]

        # 将预测特征向量和模板库中所有的特征向量计算余弦相似度，然后排序top1/top5，查看真值标签是否存在于其中
        key_list = list()
        value_list = list()
        for key, values in match_dict.items():
            for template_feats in values:
                sim = cosine_similarity(pred_feats, template_feats)
                key_list.append(key)
                value_list.append(sim)

        # 完成所有模板库匹配后计算top1/top5
        if len(value_list) == 0:
            # 模板库没有数据，检测错误
            pass
        else:
            # 先按照相似度进行排序
            sorted_idxs = list(reversed(np.argsort(value_list)))
            key_list = np.array(key_list)[sorted_idxs]
            value_list = np.array(value_list)[sorted_idxs]

            # 计算top1
            match_label = key_list[0]
            if match_label == truth_label:
                top1_num += 1
                top5_num += 1
            else:
                # 计算top5
                value_len = min(len(value_list), 5)
                if truth_label in key_list[:value_len]:
                    top5_num += 1

        # 每次完成后将预测特征向量加入模板库
        match_dict[truth_label].append(pred_feats)
        # # 如果模板库某一类别的特征数目大于指定数目，则按先进先出方式删除
        if len(match_dict[truth_label]) > N:
            match_dict[truth_label].pop(0)

    # 完成所有计算后，统计top1/top5识别率
    top1_acc = top1_num / arr_len
    top5_acc = top5_num / arr_len
    print(f'top1 num: {top1_num} - top5 num: {top5_num}')
    print(f'top1 acc: {top1_acc} - top5 acc: {top5_acc}')


if __name__ == '__main__':
    args = parse_args()
    match_path = args.match_path
    test_path = args.test_path
    N = args.N

    if match_path is not None:
        assert os.path.isfile(match_path), match_path
        match_dict = load_match_lib(match_path)
    else:
        match_dict = dict()

    assert os.path.isfile(test_path), test_path
    data_array = load_data(test_path)

    process(data_array, match_dict, N)
