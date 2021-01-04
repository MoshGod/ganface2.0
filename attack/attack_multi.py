#!/user/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .attack import Attack


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a Model with various attacks agains same images and labels.
    Arguments:
        Model (nn.Module): Model to attack.
        attacks (list): list of attacks.

    Examples::
        >>> attack1 = torchattacks.PGD(Model, eps=4/255, alpha=8/255, iters=40, random_start=False)
        >>> attack2 = torchattacks.PGD(Model, eps=4/255, alpha=8/255, iters=40, random_start=False)
        >>> attack = torchattacks.MultiAttack(Model, [attack1, attack2])
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, attacks):

        # Check validity
        ids = []
        for attack in attacks:
            ids.append(id(attack.model))

        # 如果传入的模型又不同的就打印警告
        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different Model.")

        super(MultiAttack, self).__init__("MultiAttack", attack.model)
        self.attacks = attacks
        self._attack_mode = 'only_default'

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        fails = torch.arange(images.shape[0]).to(self.device) # fails[0,1,...,10] 攻击失败的图片idx，一开始都失败
        final_images = images.clone().detach().to(self.device) # shape[10, 3, 28, 28]
        labels = labels.clone().detach().to(self.device) # shape[10,]

        for i, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])  # 把攻击失败的图片送入attack方法得到adv_imgs

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails]) # 得到预测正确的位置 [1,0,1,0,1][True, False, ..., True]
            wrongs = ~corrects # 得到预测错误的图片位置 [0,1,0,1,0]
            # succeeds 选出预测错误的，就是对抗成功的对应在原images中的下标 [4,8]
            succeeds = torch.masked_select(fails, wrongs)
            # fails中成功的fails的下标 [1,3]
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)
            # 对抗图片中对抗成功的下标选出来，赋值给原images中对应的图片
            final_images[succeeds] = adv_images[succeeds_of_fails]
            # 剩下攻击失败的继续下一次对抗 [2,4,6,8,10]
            fails = torch.masked_select(fails, corrects)

            if len(fails) == 0:
                break

        return final_images


