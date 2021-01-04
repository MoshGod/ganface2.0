#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

from .attack import Attack


class BIM(Attack):
    r"""
    简介：基本迭代方法(I-FGSM)，直接将其扩展为通过多个小步增大损失函数的变体，从而我们得到 Basic Iterative Methods（BIM）
    论文：BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
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
        >>> attack = attack_bim.BIM(Model, eps=4/255, alpha=1/255, steps=0)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=4 / 255, alpha=1 / 255, steps=0):
        super(BIM, self).__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0: # 计算一个默认的steps，从论文中得来
            self.steps = int(min(eps+ 4, 1.25 * eps))
        else:
            self.steps = steps

    def setEps(self, eps):
        self.eps = eps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach() # 记下原图

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + self.alpha * grad.sign() # 走一小步

            """下面为裁剪总步伐不超过eps 且也在[0,1]中"""
            # 小于eps的用eps，不然用alpha
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a

            # 大于eps的用eps，不然用alpha
            c = torch.clamp(ori_images + self.eps, max=1)
            images = ((b > c).float() * c + (b <= c).float() * b).detach()

        adv_images = images

        return adv_images


