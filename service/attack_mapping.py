#!/user/bin/env python    
#-*- coding:utf-8 -*-

import torch

from attack.attack_bim import BIM
from attack.attack_ffgsm import FFGSM
from attack.attack_fgsm import FGSM
from attack.attack_pgd import PGD
from attack.attack_apgd import APGD
from attack.attack_stepll import StepLL
from attack.attack_tpgd import TPGD
from attack.attack_mifgsm import MIFGSM
from attack.attack_rfgsm import RFGSM
from attack.attack_cw import CW
from attack.attack_deepfool import DeepFool


"""
img: torch.tensor [1, c, h, w]
Model: 模型
method: str, 对应对抗攻击算法的名称，一定要一模一样
eps: 对抗攻击强度
alpha:
label: 只有targeted模式会用到，torch.tensor [1]
mode: 对抗攻击模式，对应算法类里面的模式
"""


# 可能还要根据不同算法测试出来的参数细分
def attack_mapping(img, model, method, eps=0.015, label=torch.tensor([0]), mode='original'):

    # attack_img = None
    label = torch.tensor([int(label)], dtype=torch.long)

    if method == 'CW':
        attack = eval(method)(model, c=0.7, kappa=eps * 3, steps=50, lr=0.08)
    elif method == 'DeepFool':
        attack = eval(method)(model, steps=round(eps * 20))
    elif method == 'MIFGSM':
        attack = eval(method)(model, eps=eps * 0.1, decay=0.1)
    elif method not in ['FGSM', 'FFGSM']:
        attack = eval(method)(model, eps=eps * 0.7, alpha=eps / 66, steps=4)
    else:
        attack = eval(method)(model, eps=eps * 0.7)

    if mode == "targeted":
        attack.set_attack_mode(mode='targeted')

    attack_img = attack(img, label)

    return attack_img

