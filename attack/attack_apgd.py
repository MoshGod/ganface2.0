#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn
from .attack import Attack


class APGD(Attack):
    r"""
    简介：通过将网络的实际梯度估计为多个随机向量ǫ上梯度的平均值，可以获得更稳定、更有效的攻击，简称为A-PGD（average PGD）
    论文：Comment on Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network
    论文地址：[https://arxiv.org/abs/1907.00895]

    参数:
        Model (nn.Module): 被攻击的模型
        eps (float): 对于本算法，限制每一小步扰动的范围，总的不能超过eps (DEFALUT : 0.3)
        alpha (float): 每一个step对抗强度，等同于单次FGSM中的eps (DEFALUT : 2/255)
        steps (int): 攻击迭代次数 (DEFALUT : 40)
        sampling (int) : 要采样的模型数 (DEFALUT : 100)

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_apgd.APGD(Model, eps = 4/255, alpha = 8/255, steps=40, sampling=100)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, sampling=10):
        super(APGD, self).__init__("APGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.sampling = sampling

    def setEps(self, eps):
        self.eps = eps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for i in range(self.steps):

            grad = torch.zeros_like(images)
            images.requires_grad = True

            for j in range(self.sampling):

                outputs = self.model(images)
                cost = self._targeted * loss(outputs, labels).to(self.device)

                grad += torch.autograd.grad(cost, images,
                                            retain_graph=False,
                                            create_graph=False)[0]

            # grad.sign() is used instead of (grad/sampling).sign()
            adv_images = images + self.alpha * grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = images

        return adv_images