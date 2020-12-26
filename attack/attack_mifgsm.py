#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

from .attack import Attack


class MIFGSM(Attack):
    r"""
    论文：Boosting Adversarial Attacks with Momentum'
    论文地址：[https://arxiv.org/abs/1710.06081]

    参数:
        Model (nn.Module): Model to attack.
        eps (float): maximum perturbation. (DEFALUT: 8/255)
        decay (float): momentum factor. (DEFAULT: 1.0)
        steps (int): number of iterations. (DEFAULT: 5)

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_mifgsm.MIFGSM(Model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, steps=5, decay=1.0):
        super(MIFGSM, self).__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # 按行(dim=1)求grad的1-范数(p=1)
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            print(grad_norm)
            print(grad.shape)
            # Tensor.view() 重构张量维度，-1表示根据输入自动计算
            # 依据batchsize大小重构grad_norm
            print(grad_norm.view([-1] + [1] * (len(grad.shape) - 1)))
            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images