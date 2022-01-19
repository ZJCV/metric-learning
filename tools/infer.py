# -*- coding: utf-8 -*-

"""
@date: 2022/1/17 下午6:37
@file: infer.py
@author: zj
@description:
批量推理
1. 读取测试文件
2. 加载预训练模型
3. 批量测试
4. 保存训练结果（分两个文件，保存输出向量和分类概率）
"""

import os
import argparse

from tqdm import tqdm

import torch.nn.functional as F

from zcls.config.key_word import KEY_OUTPUT
# from zcls.data.build import build_data
# from zcls.model.recognizers.build import build_recognizer
from zcls.util.distributed import get_device
from zcls.util import logging

logger = logging.get_logger(__name__)

from metric.config import cfg
from metric.config.key_word import KEY_FEATS
from metric.model.build import build_model
from metric.data.build import build_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path', type=str, default=None, help='ZCls Config Path')
    parser.add_argument('--train', default=False, action='store_true', help='Use Train Data. Default: False')

    args = parser.parse_args()
    print(args)
    return args


def infer(cfg, is_train=False):
    device = get_device(0)
    # model = build_recognizer(cfg, device)
    model = build_model(cfg, device)
    model.eval()

    feats_list = list()
    logits_list = list()
    probs_list = list()
    targets_list = list()
    img_paths_list = list()

    data_loader = build_data(cfg, is_train=is_train, device_type=device.type, rank_id=0)
    for iteration, (images, targets, img_paths) in tqdm(enumerate(data_loader)):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        out_dict = model(images)

        # feats
        feats = out_dict[KEY_FEATS]
        # logits
        logits = out_dict[KEY_OUTPUT]
        # probs
        probs = F.softmax(logits, dim=1)

        feats_list.extend(feats.detach().cpu().numpy())
        logits_list.extend(logits.detach().cpu().numpy())
        probs_list.extend(probs.detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        img_paths_list.extend(img_paths)

    return feats_list, logits_list, probs_list, targets_list, img_paths_list


def save_to_csv(img_paths, data_list, csv_path):
    assert not os.path.exists(csv_path), csv_path
    assert len(img_paths) == len(data_list)

    print(f'save to {csv_path}')
    with open(csv_path, 'w') as f:
        data_len = len(img_paths)
        for idx, (img_path, items) in enumerate(zip(img_paths, data_list)):
            item_str = ','.join(items.astype(str))
            if idx < (data_len - 1):
                f.write(f'{img_path},{item_str}\n')
            else:
                f.write(f'{img_path},{item_str}')


def main():
    args = parse_args()
    if args.cfg_path:
        cfg.merge_from_file(args.cfg_path)
    cfg.freeze()
    is_train = args.train

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(args)

    feats, logits, probs, targets, img_paths = infer(cfg)

    csv_suffix = 'train' if is_train else 'test'
    feats_path = f'./outputs/{csv_suffix}_feats.csv'
    save_to_csv(img_paths, feats, feats_path)
    logits_path = f'./outputs/{csv_suffix}_logits.csv'
    save_to_csv(img_paths, logits, logits_path)
    probs_path = f'./outputs/{csv_suffix}_probs.csv'
    save_to_csv(img_paths, probs, probs_path)


if __name__ == '__main__':
    main()
