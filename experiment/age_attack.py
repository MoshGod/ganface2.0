#!/user/bin/env python    
#-*- coding:utf-8 -*- 

"""@@@
用于评估人脸年龄识别对抗攻击算法的攻击效果

需要整合成可以迭代计算预先准备好的对抗算法的模式

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from attack.attack_fgsm import *
from utils.get_dataset import *
from attack.attack_apgd import APGD # step 默认为 40
from attack.attack_tpgd import TPGD # step 默认为 7
from attack.attack_cw import CW # step 默认为 1000
from attack.attack_deepfool import DeepFool # step 默认为 3
from utils.img_utils import plot_eps_acc
from Model.model_utils import set_seed

set_seed(6666)

"""获取数据"""
batch_size = 20
root = r'D:\WorkSpace\Python\Introduction_to_Software_Engineering\dataset\Asia_age\test_4'
transform = transforms.Compose([
    # transforms.Scale(224),
    transforms.Resize((224, 224)),
    # transforms.CenterCrop((size, size)),
    # transforms.RandomRotation(0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # PIL Image → Tensor
    # transforms.Gray()
])
_, test_loader = get_Dataloader('Asia-age-4', root=root, batch_size=batch_size, transform=transform, shuffle_flag=True)


"""获取预训练模型"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model = models.inception_v3(pretrained=True).to(device)
model_save_path = '../saveModel/Asiaage_resnet_4class.pkl'
model = torch.load(model_save_path)
# 直接切换到推理模式
model.eval()


# """未攻击正确率"""
# print("True Image & Predicted Label")
#
# correct = 0
# total = len(test_loader.dataset)
#
# for images, labels in test_loader:
#     images, labels = images.to(device), labels.to(device)
#     # 前向传播
#     outputs = Model(images)
#     # max 返回(最大值,下标)， pre size: [b]
#     _, pre = torch.max(outputs.data, dim=1)
#     # 累积正确个数
#     correct += (pre == labels).sum()
#     # 展示batch_size张图片 和 其预测的类别
#     # showBatchImages(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre], False)
#
# # 预测的平均正确率
# print('Accuracy of test text: %f %%' % (100 * float(correct) / total))

"""攻击后正确率"""
# 攻击参数
epss = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# epss = [0.1,0.2,0.3] # 测试生成对抗样本的速度
# epss = np.linspace(0.01, 1, 100) #0.01
plot_eps = [0.1, 0.3, 0.5, 0.7, 0.9]

# attack_methods = [FFGSM]
# attack_methods_str = ['FFGSM']

attack_methods_with_steps = [APGD]    #, TPGD, CW, DeepFool]
attack_methods_with_steps_str = ['APGD']    #, 'TPGD', 'CW', 'DeepFool']
steps=[ ]


loss = nn.CrossEntropyLoss()

print("Attack Image & Predicted Label")

for attack_method, attack_method_str in zip(attack_methods_with_steps, attack_methods_with_steps_str):
    print(attack_method_str+' performance')
    accuracy = []
    for eps in epss:
        attack = attack_method(model, eps=eps, alpha=eps/66, steps=3)
        correct = 0
        # total = len(test_loader.dataset)
        total = batch_size * 10
        limit = 0
        flag = True
        for images, labels in test_loader:

            # print('say')
            # 获取对抗样本
            images = attack(images, labels)
            labels = labels.to(device)
            # 推理对抗样本
            outputs = model(images)
            # 获取预测值
            _, pre = torch.max(outputs.data, dim=1)
            # 计算对抗后的正确率
            # total += batch_size
            correct += (pre == labels).sum()

            # showImage(torchvision.utils.make_grid(images.cpu().data, normalize=True),
            #           [attack_method_str + ' - eps:', str(eps) + '\n'] + [pre], title_flag=True)
            # 展示batch_size张图片 和 其预测的类别
            if flag and eps in plot_eps:
                # print(eps)
                savedir = r'C:\Users\99785\Desktop\\' + attack_method_str + '\\' + str(eps) + '.png'
                show_image(torchvision.utils.make_grid(images.cpu().data, normalize=True), [attack_method_str + ' - eps:', str(eps) + '\n'] + [pre], title_flag=True)
                flag = False
            limit += 1
            if limit >= 10:
                break


        acc = 100 * float(correct) / total
        print('Accuracy of test text: %f %%' % acc,' with using eps: %f' % eps)
        accuracy.append(acc)

    # time.sleep(1)

    savedir = r'C:\Users\99785\Desktop\APGD.png'
    plot_eps_acc(attack_method_str+' performance', epss, accuracy, savedir=savedir)