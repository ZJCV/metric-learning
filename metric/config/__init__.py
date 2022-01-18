# -*- coding: utf-8 -*-

"""
@date: 2022/1/17 下午4:52
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized config.
    _C.METRIC = CN()

    return _C


cfg = add_custom_config(get_cfg_defaults())
