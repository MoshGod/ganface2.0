#!/user/bin/env python    
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .attack import Attack


class FGSM(Attack):
    r"""
    简介：快速符号梯度攻击法，通过添加由网络梯度生成的攻击噪声，基本算是对抗攻击算法的鼻祖
    论文：Explaining and harnessing adversarial examples
    论文地址：[https://arxiv.org/abs/1412.6572]

    参数:
        Model (nn.Module): 被攻击的模型
        eps (float): 对于本算法，限制每一小步扰动的范围 (默认值 : 0.007)

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_fgsm.FGSM(Model, eps=0.007)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps

    def setEps(self, eps):
        self.eps = eps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        cost = self._targeted * loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        # retain_graph是否释放计算图

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


