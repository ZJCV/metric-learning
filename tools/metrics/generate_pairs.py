# -*- coding: utf-8 -*-

"""
@date: 2022/1/19 上午11:09
@file: generate_thresh.py
@author: zj
@description: 生成全局阈值
1. 特征文件每行表示一个特征向量，每行格式如下：
    文件名,特征向量
2. 将所有特征向量载入数据集中
    2.1 采集正样本对
        2.1.1 获取每个类别数据
        2.1.2 随机打乱，折半组成正样本对
    2.2 采集负样本对
        2.2.1 获取每个类别数据
        2.2.2 随机采集100张图片，折半组成50对，去除正样本对
        2.2.3 重复2.2.2过程，直到负样本对数目达到正样本集为止
3. 负样本集标签为0，正样本集标签为1

如何判断两个文件名是否属于相同的类别，需要解析YC列表和合并列表
"""

import os
import json
import argparse
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', type=str, default=None, help='test path')
    parser.add_argument('dst_path', type=str, default=None, help='dst path')

    args = parser.parse_args()
    print(args)
    return args


def parse_json(json_path):
    assert os.path.isfile(json_path)

    with open(json_path, 'r') as f:
        data_dict = json.load(f)
        assert isinstance(data_dict, dict)

    yc_code_list = list()
    sku_name_list = list()
    for item_dict in data_dict['content']:
        assert isinstance(item_dict, dict)

        yc_code_list.append(item_dict['new'])
        sku_name_list.append(item_dict['name'])

    return yc_code_list, sku_name_list


def parse_csv(csv_path):
    assert os.path.isfile(csv_path)

    data_array = np.loadtxt(csv_path, dtype=str, delimiter=',')

    combined_data_dict = dict()
    for line in data_array:
        combined_data_dict[str(line[0])] = str(line[2])
        # combined_data_dict[str(line[0])] = str(line[1])

    return combined_data_dict


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=str)
    img_path_list = data_array[:, 0]

    # 检索出每个图像路径对应的类名
    new_img_path_list = list()
    for img_path in img_path_list:
        cls_name = os.path.split(os.path.split(img_path)[0])[1]
        new_img_path_list.append(','.join([str(img_path), str(cls_name)]))

    return new_img_path_list


def create_positive_pair(img_path_list):
    data_dict = dict()

    for img_path_str in img_path_list:
        img_path, cls_name = img_path_str.split(',')
        if cls_name not in data_dict.keys():
            data_dict[cls_name] = list()
        data_dict[cls_name].append(img_path)

    positive_pair_list = list()
    for key, value in data_dict.items():
        value_len = len(value)
        if value_len < 2:
            continue
        half_value_len = value_len // 2

        random.shuffle(value)
        # 正样本对的标签为1
        tmp_pair_list = [(x, y, 1) for x, y in zip(value[:half_value_len], value[half_value_len:])]
        positive_pair_list.extend(tmp_pair_list)

    return positive_pair_list


def create_negative_pair(img_path_list, positive_pair_num):
    negative_pair_list = list()

    epoch = 0
    while len(negative_pair_list) < positive_pair_num:
        # 打乱数据
        random.shuffle(img_path_list)

        # 取前100个图像
        tmp_list = img_path_list[:100]
        # 组成50个图像对
        tmp_pair_list = [(x, y) for x, y in zip(tmp_list[:50], tmp_list[50:])]
        # 遍历图像对列表，过滤正样本对
        valid_num = 0
        for tmp_pair in tmp_pair_list:
            x, y = tmp_pair
            x_img, x_cls = x.split(',')
            y_img, y_cls = y.split(',')
            if x_cls == y_cls:
                continue

            # 负样本对的标签为0
            negative_pair_list.append([x_img, y_img, 0])
            valid_num += 1

        # 对样本对列表进行排序，过滤相似的结果

        print(f'epoch: {epoch}, valid num: {valid_num}')
        epoch += 1

    return negative_pair_list


def process(img_path_list):
    positive_pair_list = create_positive_pair(img_path_list)
    negative_pair_list = create_negative_pair(img_path_list, len(positive_pair_list))

    total_pair_list = list()
    total_pair_list.extend(positive_pair_list)
    total_pair_list.extend(negative_pair_list)

    return total_pair_list


def save_to_csv(pair_list, dst_path):
    assert not os.path.isfile(dst_path), dst_path

    list_len = len(pair_list)
    with open(dst_path, 'w') as f:
        for idx, (img1, img2, label) in enumerate(pair_list):
            if idx < (list_len - 1):
                f.write(f"{img1} {img2} {label}\n")
            else:
                f.write(f"{img1} {img2} {label}")


if __name__ == '__main__':
    args = parse_args()
    test_path = args.test_path
    dst_path = args.dst_path

    if not os.path.isfile(test_path):
        raise ValueError(f'{test_path} does not exists')
    if os.path.isfile(dst_path):
        raise ValueError(f'{dst_path} does not exists')

    print('load data ...')
    img_path_list = load_data(test_path)

    print('process ...')
    pair_list = process(img_path_list)

    print(f'save to {dst_path}')
    save_to_csv(pair_list, dst_path)
