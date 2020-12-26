#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

from .attack import Attack


class StepLL(Attack):
    r"""
    StepLL，也叫ILCM(Least-Likely-Class Iterative Methods) 迭代最小可能类法
    简介：通过用识别概率最小的类别（目标类别）代替对抗扰动中的类别变量，再将原始图像减去该扰动，原始图像就变成了对抗样本，并能输出目标类别。
    论文：Adversarial Examples in the Physical World
    论文地址：[https://arxiv.org/abs/1607.02533]

    参数:
        Model (nn.Module): 被攻击的模型
        eps (float): 对于本算法，限制每一小步扰动的范围，总的不能超过eps (默认值: 4/255)
        alpha (float): 每一个step对抗强度，等同于单次FGSM中的eps (DEFALUT : 1/255)
        steps (int): 攻击迭代次数 (DEFALUT : 0)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: `(N, C, H, W)`
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_stepll.StepLL(Model, eps=4/255, alpha=1/255, steps=0)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=4 / 255, alpha=1 / 255, steps=0):
        super(StepLL, self).__init__("StepLL", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def setEps(self, eps):
        self.eps = eps

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)

        outputs = self.model(images)
        _, labels = torch.min(outputs.data, dim=1)  # 概率最小的类别作为标签
        labels = labels.detach_()

        loss = nn.CrossEntropyLoss()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images - self.alpha * grad.sign()  # 让预测结果朝着这个ll类走

            # 下面这里是为了把 FGSM里的一步eps分成很多小步的alpha
            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (a > adv_images).float() * a
            c = (b > images + self.eps).float() * (images + self.eps) + (images + self.eps >= b).float() * b
            images = torch.clamp(c, max=1).detach()

        adv_images = images

        return adv_images
