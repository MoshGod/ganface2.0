#!/user/bin/env python    
#-*- coding:utf-8 -*-

import numpy as np
import json
import os
import sys
import time
import pprint
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from PIL import Image
from torchvision import models
from torch.nn import functional as F
from attack.attack_bim import BIM
from attack.attack_ffgsm import FFGSM
from attack.attack_fgsm import FGSM
from attack.attack_pgd import PGD
from attack.attack_apgd import APGD
from attack.attack_rfgsm import RFGSM
from attack.attack_stepll import StepLL
from attack.attack_tpgd import TPGD
from attack.attack_mifgsm import MIFGSM
from Model.model_utils import set_seed
from service.attack_mapping import attack_mapping
from matplotlib import pyplot as plt


pp = pprint.PrettyPrinter(indent=2)
set_seed(6666)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

json_data_path = r'../data/class_index.json'
data_mode = 'imagenet'
idx_to_class = json.load(open(json_data_path))[data_mode]
idx2label = [idx_to_class[str(k)][1] for k in range(len(idx_to_class))]

transform1 = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform2 = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
])


# class Normalize(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalize, self).__init__()
#         self.register_buffer('mean', torch.Tensor(mean))
#         self.register_buffer('std', torch.Tensor(std))
#
#     def forward(self, input):
#         # Broadcasting
#         mean = self.mean.reshape(1, 3, 1, 1)
#         std = self.std.reshape(1, 3, 1, 1)
#         return (input - mean) / std

#
# norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Model = nn.Sequential(
#     norm_layer,
#     models.resnet18(pretrained=True)
# ).to(device)

model_save_path = '../saveModel/resnet_18.pkl'
# torch.save(Model, model_save_path)

# Model = torch.load(model_save_path)
model = models.resnet18(pretrained=True).to(device)
torch.save(model, model_save_path)
model = model.eval()


attacks = [FGSM(model, eps=8/255),
           BIM(model, eps=8/255, alpha=2/255, steps=7),
           # RFGSM(Model, eps=8/255, alpha=4/255, steps=1),
           PGD(model, eps=8/255, alpha=2/255, steps=7),
           FFGSM(model, eps=8/255, alpha=12/255),
           TPGD(model, eps=8/255, alpha=2/255, steps=7),
           MIFGSM(model, eps=8/255, decay=1.0, steps=5)
           ]

# for attack in attacks:
#     print("-" * 70)
#     print(attack)
#
#     start = time.time()
#     images = attack(images, labels)
#     labels = labels.to(device)
#     outputs = Model(images)
#
#     _, pre = torch.max(outputs.data, 1)
#
#     # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [idx2label[pre.cpu().item()]])
#
#     print('Total elapsed time (sec) : %.2f' % (time.time() - start))


def predict(data):
    img_np = data.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()

    data = torch.unsqueeze(data, 0).to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        # print('the predict class(idx): {}({})'.format(idx_to_class[str(pred.item())], pred.item()))

    predict_detail = {'class':'', 'index':'', 'probs':{}}
    print('The probs detail: ')
    # print(pred.shape)
    values, indices = probs.topk(5, sorted=True)
    for value, indice in zip(values[0], indices[0]):
        key = idx_to_class[str(indice.cpu().item())][1]
        value = value.cpu().item()
        predict_detail['probs'][key] = round(value*100, 2)
        # print(idx_to_class[str(i)].rjust(3)+':','{:.6f}'.format(value))

    predict_class = idx_to_class[str(pred.item())]
    predict_detail['index'] = str(pred.item())
    predict_detail['class'] = predict_class[1]
    pp.pprint(predict_detail)
    return predict_detail # 返回预测的概率和 对抗样本


def generate(img, label, method, eps, mode='original'):
    img = torch.unsqueeze(img, 0).to(device)  # 输入到attack里要先增加一个维度
    img = attack_mapping(img, model, method=method, eps=eps, label=label, mode=mode)
    img = torch.squeeze(img)
    return img


if __name__ == '__main__':
    path = r'C:\Users\99785\Desktop\dog.jpg'
    const_image = Image.open(path)

    data = transform1(const_image)

    # data[0] = data[0] * 0.229 + 0.485
    # data[1] = data[1] * 0.224 + 0.456
    # data[2] = data[2] * 0.225 + 0.406
    # img2np = data.permute(1, 2, 0).cpu().numpy()
    # plt.imshow(img2np)
    # plt.show()

    # label = torch.tensor([388], dtype=torch.long)
    ori_detail = predict(data)
    # attacks[0](Model)
    data = transform2(const_image)
    # adv_img = generate(data, ori_detail['index'], 'CW', 1, mode='targeted')
    adv_img = generate(data, '888', 'CW', 1, mode='targeted')
    adv_detail = predict(adv_img)

    # adv_img[0] = adv_img[0] * 0.229 + 0.485
    # adv_img[1] = adv_img[1] * 0.224 + 0.456
    # adv_img[2] = adv_img[2] * 0.225 + 0.406
    # img2np = adv_img.permute(1, 2, 0).cpu().numpy()
    # plt.imshow(img2np)
    # plt.show()
    # print("Adversarial Image & Predicted Label")
