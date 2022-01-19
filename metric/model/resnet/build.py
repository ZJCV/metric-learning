# -*- coding: utf-8 -*-

"""
@date: 2022/1/18 下午7:16
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet18, resnet34, resnet50

from zcls.config.key_word import KEY_OUTPUT
from metric.config.key_word import KEY_FEATS


class ResNet(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 arch='resnet18'
                 ) -> None:
        super().__init__()

        assert arch in ['resnet18', 'resnet34', 'resnet50']
        self.model = eval(arch)(num_classes=1000, pretrained=True)

        if num_classes != 1000:
            in_features = self.model.fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.fc = new_fc

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        feats = torch.flatten(x, 1)
        x = self.model.fc(feats)

        return x, feats

    def forward(self, x: Tensor) -> dict:
        x, feats = self._forward_impl(x)

        return {
            KEY_OUTPUT: x,
            KEY_FEATS: feats
        }


def get_resnet(num_classes=1000, arch='resnet18'):
    return ResNet(num_classes=num_classes, arch=arch)
