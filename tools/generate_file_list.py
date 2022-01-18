# -*- coding: utf-8 -*-

"""
@date: 2022/1/17 下午3:53
@file: generate_file_list.py
@author: zj
@description:
CUB_200_2011数据集位于/path/to/data路径下
.
├── data
│   └── cub -> /home/zj/data/CUB_200_2011/CUB_200_2011
├── main.py
└── tools
    ├── generate_file_list.py
    └── __init__.py
"""

import os

import numpy as np
from zcls.config.key_word import KEY_SEP


def load_label_class(cls_path):
    assert os.path.isfile(cls_path), cls_path

    cls_array = np.loadtxt(cls_path, delimiter=' ', dtype=str)

    # label_class_dict = dict()
    # for label, cls in cls_array:
    #     label_class_dict[label] = cls
    #
    # return cls_array

    return cls_array[:, 1]


def load_train_test(train_test_split_path, images_path, image_class_labels_path, image_prefix=''):
    assert os.path.isfile(train_test_split_path), train_test_split_path
    assert os.path.isfile(images_path), images_path
    assert os.path.isfile(image_class_labels_path), image_class_labels_path

    train_test_split_array = np.loadtxt(train_test_split_path, delimiter=' ', dtype=str)
    image_array = np.loadtxt(images_path, delimiter=' ', dtype=str)
    image_class_label_array = np.loadtxt(image_class_labels_path, delimiter=' ', dtype=str)
    assert len(train_test_split_array) == len(image_array) == len(image_class_label_array)

    train_dict = dict()
    test_dict = dict()

    train_num = 0
    test_num = 0
    for (_, is_train), (_, img_path), (_, label) in zip(train_test_split_array, image_array, image_class_label_array):
        img_path = os.path.join(image_prefix, img_path)
        if int(is_train) == 1:
            if label not in train_dict.keys():
                train_dict[label] = list()
            train_dict[label].append(img_path)
            train_num += 1
        else:
            if label not in test_dict.keys():
                test_dict[label] = list()
            test_dict[label].append(img_path)
            test_num += 1

    return train_dict, test_dict, train_num, test_num


def save_to_cls(cls_array, dst_path):
    assert not os.path.isfile(dst_path), dst_path

    np.savetxt(dst_path, cls_array, fmt='%s', delimiter=' ')


def save_to_csv(data_dict, dst_path):
    """
    注意：配置文件中标签从1开始，训练中label从0开始
    """
    assert not os.path.exists(dst_path), dst_path

    data_list = list()
    for key, values in data_dict.items():
        for img_path in values:
            data_list.append([img_path, key])

    data_len = len(data_list)
    with open(dst_path, 'w') as f:
        for idx, (img_path, label) in enumerate(data_list):
            label = int(label) - 1

            if idx < (data_len - 1):
                f.write(f"{img_path}{KEY_SEP}{label}\n")
            else:
                f.write(f"{img_path}{KEY_SEP}{label}")


if __name__ == '__main__':
    cls_path = './data/cub/classes.txt'
    cls_array = load_label_class(cls_path)

    dst_cls_path = f'./data/cub/cls_{len(cls_array)}.csv'
    save_to_cls(cls_array, dst_cls_path)

    image_prefix = './data/cub/images'

    train_test_split_path = './data/cub/train_test_split.txt'
    images_path = './data/cub/images.txt'
    image_class_labels_path = './data/cub/image_class_labels.txt'
    train_dict, test_dict, train_num, test_num = load_train_test(train_test_split_path,
                                                                 images_path,
                                                                 image_class_labels_path,
                                                                 image_prefix)

    train_path = f'./data/cub/train_{train_num}.csv'
    save_to_csv(train_dict, train_path)
    test_path = f'./data/cub/test_{test_num}.csv'
    save_to_csv(test_dict, test_path)
