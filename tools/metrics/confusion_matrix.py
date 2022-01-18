# -*- coding: utf-8 -*-

"""
@date: 2022/1/17 下午7:32
@file: confusion_matrix.py
@author: zj
@description: 混淆矩阵计算
1. 加载测试文件
2.
"""

import os

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix, \
    auc, roc_curve, roc_auc_score, \
    top_k_accuracy_score


def load_data(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_array = np.loadtxt(csv_path, delimiter=',', dtype=float)

    truth_list = list()
    pred_label_list = list()
    pred_probs_list = list()

    for item in data_array:
        label = int(item[0])
        truth_list.append(label)

        pred_label = np.argmax(item[1:])
        pred_label_list.append(pred_label)

        pred_probs_list.append(item[1:])

    return np.array(truth_list), np.array(pred_label_list), np.array(pred_probs_list)


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


def score_roc(y_pred_prob, y_test):
    print('roc curve ...')
    # print('y_test[:10]:', y_test[:10])
    # print('y_pred_prob shape:', y_pred_prob.shape)
    thresh_list = list()

    uni_list = np.unique(y_test)
    # print(uni_list)
    # 基于每个类别进行二分类ROC曲线绘制，获取最佳阈值
    for i in uni_list:
        y_pred = y_pred_prob[:, i]
        y_test_new = y_test == i
        y_test_new = y_test_new.astype(int)
        # 计算ROC曲线
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_new, y_pred, pos_label=1)
        # print('tpr: {}'.format(tpr))
        # print('fpr: {}'.format(fpr))
        # print('thresholds: {}'.format(thresholds))

        # 计算ROC曲线下面积
        # the area under ROC curve
        # print('y_pred shape:', y_pred.shape, ' y_test_new shape:', y_test_new.shape)
        auc_score = roc_auc_score(y_test_new, y_pred)
        roc_auc = auc(fpr, tpr)
        # print('auc:', auc_score, ' roc_auc:', roc_auc)
        assert auc_score == roc_auc

        # # 显示ROC曲线
        # # show ROC curve
        # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        # display.plot()
        # plt.show()

        # 计算最佳阈值
        # compute best threshold
        thresh = thresholds[np.argmax(tpr - fpr)]
        # print('thresh:', thresh)
        thresh_list.append(thresh)

    return thresh_list


if __name__ == '__main__':
    csv_path = './outputs/probs.csv'
    truths, pred_labels, pred_probs = load_data(csv_path)

    top_5_acc = top_k_accuracy_score(truths, pred_probs, k=5)
    top_5_acc_num = top_k_accuracy_score(truths, pred_probs, k=5, normalize=False)
    print('top_5_acc:', top_5_acc, ' top_5_acc_num:', top_5_acc_num)

    score(pred_labels, truths)
    thresh_list = score_roc(pred_probs, truths)
    # print(pred_probs[:10])
    # print(thresh_list)
    assert len(pred_probs[0]) == len(thresh_list)

    # 使用最佳阈值对每个预测概率进行过滤，然后再计算分类标签
    tmp_array = np.array(pred_probs) >= np.array(thresh_list)
    pred_probs_new = pred_probs * tmp_array
    pred_labels_new = np.argmax(pred_probs_new, axis=1)
    print(pred_labels_new)

    score(pred_labels_new, truths)

    top_5_acc = top_k_accuracy_score(truths, pred_probs_new, k=5)
    top_5_acc_num = top_k_accuracy_score(truths, pred_probs_new, k=5, normalize=False)
    print('top_5_acc:', top_5_acc, ' top_5_acc_num:', top_5_acc_num)