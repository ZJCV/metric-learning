# -*- coding: utf-8 -*-

"""
@date: 2022/1/19 下午4:38
@file: compute_roc_threshold.py
@author: zj
@description: 
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix, auc, roc_curve, roc_auc_score, RocCurveDisplay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_path', type=str, default=None, help='pair path')
    parser.add_argument('test_path', type=str, default=None, help='test path')

    args = parser.parse_args()
    print(args)
    return args


def load_pairs(csv_path):
    assert os.path.isfile(csv_path), csv_path

    pair_array = np.loadtxt(csv_path, delimiter=' ', dtype=str)
    return pair_array


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=str)
    data_dict = dict()
    for item in data_array:
        img_path = str(item[0])
        feats = np.array(item[1:], dtype=float)

        data_dict[img_path] = feats

    return data_dict


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


def compute_probs(pair_array, data_dict):
    probs_list = list()
    labels_list = list()

    for img1, img2, label in pair_array:
        feats_1 = data_dict[img1]
        feats_2 = data_dict[img2]

        prob = cosine_similarity(feats_1, feats_2)
        probs_list.append(prob)
        labels_list.append(int(label))

    return probs_list, labels_list


def score(y_pred, y_test):
    print('score ...')
    print('y_pred[:10]:', y_pred[:10])
    print('y_test[:10]:', y_test[:10])

    # 计算准确率
    # accuracy
    acc = accuracy_score(y_test, y_pred)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)
    print('acc:', acc, ' acc_num:', acc_num)

    # 计算精度
    # precision
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_micro = precision_score(y_test, y_pred, average='micro')
    print('precision_macro:', precision_macro, ' precision_micro:', precision_micro)

    # 计算召回率
    # recall
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    print('recall_macro:', recall_macro, ' recall_micro:', recall_micro)

    # 混淆矩阵
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize=None)
    print(cm)


def score_roc(y_pred, y_test):
    print('score roc ...')
    print('y_pred[:10]:', y_pred[:10])
    print('y_test[:10]:', y_test[:10])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    # print('tpr: {}'.format(tpr))
    # print('fpr: {}'.format(fpr))
    # print('thresholds: {}'.format(thresholds))

    # 计算ROC曲线下面积
    # the area under ROC curve
    # print('y_pred shape:', y_pred.shape, ' y_test_new shape:', y_test_new.shape)
    auc_score = roc_auc_score(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # print('auc:', auc_score, ' roc_auc:', roc_auc)
    assert auc_score == roc_auc

    # 显示ROC曲线
    # show ROC curve
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    # plt.show()
    plt.savefig('./outputs/roc.jpg')

    # 计算最佳阈值
    # compute best threshold
    thresh = thresholds[np.argmax(tpr - fpr)]

    return thresh


if __name__ == '__main__':
    args = parse_args()
    pair_path = args.pair_path
    test_path = args.test_path

    assert os.path.isfile(pair_path), pair_path
    assert os.path.isfile(test_path), test_path

    pair_array = load_pairs(pair_path)
    data_dict = load_data(test_path)

    probs_list, labels_list = compute_probs(pair_array, data_dict)

    pred_labels_list = np.array(np.array(probs_list) > 0.5, dtype=int)
    score(pred_labels_list, labels_list)

    best_th = score_roc(probs_list, labels_list)
    print(f'best threshold: {best_th}')

    pred_labels_list = np.array(np.array(probs_list) > best_th, dtype=int)
    score(pred_labels_list, labels_list)
