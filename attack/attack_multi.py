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
        fails = torch.arange(images.shape[0]).to(self.device) # [0,1,..,batch]
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        for i, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])  # 把image[fails]送入attack方法得到adv_imgs

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails]) # 得到攻击成功的数量
            wrongs = ~corrects #

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)

            if len(fails) == 0:
                break

        return final_images


