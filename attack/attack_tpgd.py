#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attack import Attack


class TPGD(Attack):
    r"""
    简介：基于KL-Divergence损失函数的PGD对抗攻击算法
    论文：Theoretically Principled Trade-off between Robustness and Accuracy
    论文地址：[https://arxiv.org/abs/1901.08573]
    KL-Divergence loss：[https://blog.csdn.net/u012223913/article/details/75112246], [https://zhuanlan.zhihu.com/p/45200767]

    Tips：原论文第26页截取
    Therefore, we use FGSMk with the cross-entropy
    loss to calculate the adversarial example X0
    in the regularization term, and the perturbation step size η1 and
    number of iterations K are the same as in the beginning of Section

    参数:
        model (nn.Module): 被攻击的模型
        eps (float): 对于本算法，限制每一步扰动的范围 (默认值 : 8/255)
        alpha (float): 每一步对抗强度，等同于单次FGSM中的eps. (默认值 : 2/255)
        steps (int): 攻击迭代次数. (默认值 : 7)

    使用示例:
        >>> attack = attack_tpgd.TPGD(Model, eps=8/255, alpha=2/255, steps=7)
        >>> adv_images = attack(images)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=7):
        super(TPGD, self).__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)

        # torch.randn_like(input)  生成形状和input相同的Tensor，以均值为0方差为1的数据填充
        adv_images = images.clone().detach() + 0.001 * torch.randn_like(images).to(self.device).detach()  # 添加随机扰动
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # 裁剪扰动范围

        loss = nn.KLDivLoss(reduction='sum')

        for i in range(self.steps):
            adv_images.requires_grad = True
            logit_ori = self.model(images)
            logit_adv = self.model(adv_images)

            cost = self._targeted * loss(F.log_softmax(logit_adv, dim=1),
                                         F.softmax(logit_ori, dim=1)).to(self.device)
            # retain_graph - 是否保留计算图，create_graph - 需要计算高阶导数时必须为True
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
