#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import torch
from .attack import Attack


class DeepFool(Attack):
    r"""

    DeepFool方法是基于梯度迭代方法中生成扰动最小的，并且能有较高的攻击准确率
    论文：DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks
    论文地址：[https://arxiv.org/abs/1511.04599]

    参数:
        Model (nn.Module): 被攻击的模型
        steps (int): 迭代的次数 (默认值 : 3)

    Shape:
        - images: (N, C, H, W)
             N = batchsize
             C = 通道数
             H, W = 图片尺寸
            必须在[0,1]范围内
        - labels: (N) 图片对应类别的下标
        - output: (N, C, H, W)   同images

    使用示例:
        >>> attack = attack_deepfool.DeepFool(Model, steps=3)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, steps=3):
        super(DeepFool, self).__init__("DeepFool", model)
        self.steps = steps

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)

        for b in range(images.shape[0]):  # 遍历 batchsize

            image = images[b:b + 1, :, :, :]

            image.requires_grad = True
            output = self.model(image)[0]

            _, pre_0 = torch.max(output, 0)
            f_0 = output[pre_0]
            grad_f_0 = torch.autograd.grad(f_0, image,
                                           retain_graph=False,
                                           create_graph=False)[0]
            num_classes = len(output)

            for i in range(self.steps):
                image.requires_grad = True
                output = self.model(image)[0]
                _, pre = torch.max(output, 0)

                if pre != pre_0:
                    image = torch.clamp(image, min=0, max=1).detach()
                    break

                r = None
                min_value = None

                for k in range(num_classes):  # 基于分类个数
                    if k == pre_0:
                        continue

                    f_k = output[k]
                    grad_f_k = torch.autograd.grad(f_k, image,
                                                   retain_graph=True,
                                                   create_graph=True)[0]

                    f_prime = f_k - f_0
                    grad_f_prime = grad_f_k - grad_f_0
                    value = torch.abs(f_prime) / torch.norm(grad_f_prime)

                    if r is None:
                        r = (torch.abs(f_prime) / (torch.norm(grad_f_prime) ** 2)) * grad_f_prime
                        min_value = value
                    else:
                        if min_value > value:
                            r = (torch.abs(f_prime) / (torch.norm(grad_f_prime) ** 2)) * grad_f_prime
                            min_value = value

                image = torch.clamp(image + r, min=0, max=1).detach()

            images[b:b + 1, :, :, :] = image

        adv_images = images

        return adv_images
