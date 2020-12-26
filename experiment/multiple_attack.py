#!/user/bin/env python
#-*- coding:utf-8 -*-

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from attack.attack_multi import MultiAttack
from attack.attack_pgd import PGD
from Model.model import ConvModel

from utils.get_dataset import get_mnist
from Model import *


batch_size = 128


test_ds, test_loader = get_mnist(is_train=False, batch_size=batch_size, shuffle=False)

model = ConvModel(28, 3, 10)
model.load_state_dict(torch.load("./Model/target.pth"))
model = model.eval()


pgd = PGD(model, eps=4/255, alpha=2/255, steps=4, random_start=True)
atk = MultiAttack([pgd]*10)

# Number of Random Restart = 1
pgd.save(data_loader=test_loader, save_path='../data/temp.pt', verbose=True)

# Number of Random Restart = 10
atk.save(data_loader=test_loader, save_path='temp.pt', verbose=True)

