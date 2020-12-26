#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn
import warnings

import torch.optim as optim

from .attack import Attack


class CW(Attack):
    r"""
    使用论文中的 CW(L2) 攻击方法
    论文：Towards Evaluating the Robustness of Neural Networks
    论文地址：[https://arxiv.org/abs/1608.04644]


    参数:
        Model (nn.Module): 被攻击的模型
        targeted (bool): True - 将图片沿着给定标签的方向修改
                         False  - 将图片沿着偏离原标签的方向修改 (默认值: False)
        c (float): 用于box-constraint的参数. (默认值: 1e-4)
            `
        kappa (float): 论文中f(x)函数的下限. (默认值 : 0)

        steps (int): 攻击迭代的次数 (默认值 : 1000)
        lr (float): 学习率 (默认值 : 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    Examples::
        >>> attack = attack_cw.CW(Model, targeted=False, c=1e-4, kappa=0, steps=1000, lr=0.1)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, targeted=False, c=0.7, kappa=0, steps=50, lr=0.08):
        super(CW, self).__init__("CW", model)
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        if self._attack_mode == 'targeted':
            self.targeted = True

        print(self._attack_mode)

        images = images.to(self.device)
        labels = labels.to(self.device)

        # f-function in the paper
        def f(x):
            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            # If targeted, optimize for making the other class most likely
            if self.targeted:
                return torch.clamp(i - j, min=-self.kappa)

            # If untargeted, optimize for making the other class most likely
            else:
                return torch.clamp(j - i, min=-self.kappa)

        w = torch.zeros_like(images).to(self.device)
        w.detach_()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)
        prev = 1e10

        for step in range(self.steps):

            a = 1 / 2 * (nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(self.c * f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.steps // 10) == 0:
                if cost > prev:
                    warnings.warn("Early Stopped cause the loss did not converged.")
                    return (1 / 2 * (nn.Tanh()(w) + 1)).detach()
                prev = cost

            # print('- CW Attack Progress : %2.2f %%        ' %((step+1)/self.steps*100), end='\r')

        adv_images = (1 / 2 * (nn.Tanh()(w) + 1)).detach()

        return adv_images
