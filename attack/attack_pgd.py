#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

from .attack import Attack


class PGD(Attack):
    r"""
    论文：Towards Deep Learning Models Resistant to Adversarial Attacks
    论文地址：[https://arxiv.org/abs/1706.06083]

    参数:
        Model (nn.Module): 被攻击的模型
        eps (float): 对于本算法，限制每一小步扰动的范围，总的不能超过eps (DEFALUT : 0.3)
        alpha (float): 每一个step对抗强度，等同于单次FGSM中的eps (DEFALUT : 2/255)
        steps (int): 攻击迭代次数 (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)
        targeted (bool): using targeted attack with input labels as targeted labels. (DEFAULT : False)

    Shape:
        - images: `(N, C, H, W)`
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_pgd.PGD(Model, eps = 4/255, alpha = 8/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False, targeted=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def setEps(self, eps):
        self.eps = eps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()
        # 返回的tensor和原tensor在梯度上或者数据上没有任何关系
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            # 将输入input张量每个元素的范围限制到区间[min, max]，返回结果到一个新张量
            # 将每一小步的扰动限制在[-eps, eps]范围之内
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # 添加扰动后的样本像素限制在[0, 1]范围之内
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    '''
    tensor.clone() & tensor.detach():
    https://zhuanlan.zhihu.com/p/148061684
    https://blog.csdn.net/weixin_43199584/article/details/106876679

    tensor.clone()
    返回tensor的拷贝，返回的新tensor和原来的tensor具有同样的大小和数据类型。
    原tensor的requires_grad=True
    clone()返回的tensor是中间节点，梯度会流向原tensor，即返回的tensor的梯度会叠加在原tensor上

    tensor.detach()
    返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False.
    修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存.
    但如果对其中一个tensor执行某些内置操作，则会报错，例如resize_、resize_as_、set_、transpose_.

    总结：
    torch.detach() — 新的tensor会脱离计算图，不会牵扯梯度计算
    torch.clone() — 新的tensor充当中间变量，会保留在计算图中，参与梯度计算（回传叠加），但是一般不会保留自身梯度
    '''
