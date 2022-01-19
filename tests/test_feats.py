# -*- coding: utf-8 -*-

"""
@date: 2022/1/18 下午7:38
@file: test_feats.py
@author: zj
@description: 
"""

import torch

from zcls.config.key_word import KEY_OUTPUT

from metric.config import cfg
from metric.config.key_word import KEY_FEATS
from metric.model.build import build_model

if __name__ == '__main__':
    cfg_file = 'tests/r18.yaml'
    cfg.merge_from_file(cfg_file)

    device = torch.device('cpu')
    model = build_model(cfg, device)

    print(model)

    data = torch.randn(1, 3, 224, 224)
    out_dict = model(data)

    feats = out_dict[KEY_FEATS]
    outputs = out_dict[KEY_OUTPUT]

    print(feats.shape)
    print(outputs.shape)
