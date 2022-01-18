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

from tqdm import tqdm

import torch
import torch.nn.functional as F

from zcls.config.key_word import KEY_OUTPUT
from zcls.data.build import build_data
from zcls.model.recognizers.build import build_recognizer
from zcls.util.distributed import init_distributed_training, get_device, get_local_rank, synchronize
from zcls.util.parser import parse_args, load_config
from zcls.util import logging

logger = logging.get_logger(__name__)

from metric.config import cfg


def infer(cfg, is_train=False):
    device = get_device(0)
    model = build_recognizer(cfg, device)
    model.eval()

    logits_list = list()
    probs_list = list()
    targets_list = list()

    data_loader = build_data(cfg, is_train=is_train, device_type=device.type, rank_id=0)
    for iteration, (images, targets) in tqdm(enumerate(data_loader)):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # logits
        logits = model(images)[KEY_OUTPUT]
        # probs
        probs = F.softmax(logits, dim=1)

        logits_list.extend(logits.detach().cpu().numpy())
        probs_list.extend(probs.detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

    return logits_list, probs_list, targets_list


def save_to_csv(targets, data_list, csv_path):
    assert not os.path.exists(csv_path), csv_path
    assert len(targets) == len(data_list)

    print(f'save to {csv_path}')
    with open(csv_path, 'w') as f:
        data_len = len(targets)
        for idx, (label, items) in enumerate(zip(targets, data_list)):
            item_str = ','.join(items.astype(str))
            if idx < (data_len - 1):
                f.write(f'{label},{item_str}\n')
            else:
                f.write(f'{label},{item_str}')


def main():
    args = parse_args()
    load_config(args, cfg)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(args)

    is_train = True
    logits, probs, targets = infer(cfg)

    csv_suffix = 'train' if is_train else 'test'
    logits_path = f'./outputs/{csv_suffix}_logits.csv'
    save_to_csv(targets, logits, logits_path)
    probs_path = f'./outputs/{csv_suffix}_probs.csv'
    save_to_csv(targets, probs, probs_path)


if __name__ == '__main__':
    main()
