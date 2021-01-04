#!/user/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .attack import Attack


class GN(Attack):
    r"""
    简介：增加高斯噪声，最基本基本基本的思想
    参数:
        model (nn.Module): Model to attack.
        sigma (nn.Module): sigma (DEFAULT: 0.1).

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = torchattacks.GN(Model)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, sigma=0.1):
        super(GN, self).__init__("GN", model)
        self.sigma = sigma
        self._attack_mode = 'only_default'

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        # randn_like返回大小为input的size，由均值为0，方差为1的正态分布的随机数值
        adv_images = images + self.sigma * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


