# -*- coding: utf-8 -*-

"""
@date: 2022/1/19 下午2:41
@file: collate_fn.py
@author: zj
@description: 
"""

import torch

import numpy as np


def collate_fn(data):
    """
    :param data: a list of tuples with (image, label, img_path)
    :return:
    """
    images = list()
    labels = list()
    img_paths = list()

    for (image, label, img_path) in data:
        images.append(image)
        labels.append(label)
        img_paths.append(img_path)

    return torch.stack(images), torch.from_numpy(np.array(labels, dtype=int)), img_paths
