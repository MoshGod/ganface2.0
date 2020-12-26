#!/user/bin/env python    
#-*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from attack.attack_fgsm import FGSM  # 快
from attack.attack_bim import BIM  # 有个step
from attack.attack_ffgsm import FFGSM  # 快
from attack.attack_stepll import StepLL  # step
from attack.attack_pgd import PGD  # step 默认为 40
from attack.attack_apgd import APGD  # step 默认为 40
from attack.attack_tpgd import TPGD  # step 默认为 7
from attack.attack_cw import CW  # step 默认为 1000
from attack.attack_deepfool import DeepFool # step 默认为 3
from utils.get_dataset import get_mnist, show_batch_images, show_image
from utils.img_utils import plot_eps_acc
from Model.model_utils import set_seed

set_seed(6666)

batch_size = 32
# train_ds, train_loader = get_mnist(batch_size, True)
test_ds, test_loader = get_mnist(batch_size, False)
classes = test_ds.classes

"""获取预训练模型"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model = models.inception_v3(pretrained=True).to(device)
model_save_path = '../saveModel/mnist_1_0.pkl'
model = torch.load(model_save_path)
# 直接切换到推理模式
model.eval()


# """未攻击正确率"""
print("True Image & Predicted Label")

correct = 0
total = 0
flag = True
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    # 前向传播
    outputs = model(images)
    # max 返回(最大值,下标)， pre size: [b]
    _, pre = torch.max(outputs.data, dim=1)
    # 迭代次数+1
    total += batch_size
    # 累积正确个数
    correct += (pre == labels).sum()
    if flag:
        flag = False
        # 展示batch_size张图片 和 其预测的类别
        savedir = r'C:\Users\99785\Desktop\APGD\origin.png'
        show_image(torchvision.utils.make_grid(images.cpu().data, normalize=True), ['Origin', ' - eps: 0\n', pre], title_flag=True)

# 预测的平均正确率
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


"""攻击后正确率"""
# 攻击参数
epss = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# epss = [0.1,0.2,0.3] # 测试生成对抗样本的速度
# epss = np.linspace(0.01, 1, 100) #0.01
plot_eps = [0.1, 0.3, 0.5, 0.7, 0.9]

# attack_methods = [FFGSM]
# attack_methods_str = ['FFGSM']

# attack_methods_with_steps = [APGD, TPGD, CW, DeepFool]
# attack_methods_with_steps_str = ['APGD', 'TPGD', 'CW', 'DeepFool']


attack_methods_with_steps = [DeepFool]
attack_methods_with_steps_str = ['DeepFool']

steps = [1, 2, 3, 4, 5, 7, 9, 10, 15, 20, 25]
# kappas = [0, 0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]
# cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

loss = nn.CrossEntropyLoss()

print("Attack Image & Predicted Label")

for attack_method, attack_method_str in zip(attack_methods_with_steps, attack_methods_with_steps_str):
    print(attack_method_str+' performance')
    accuracy = []
    for step in steps:
        attack = attack_method(model, steps=step)
        correct = 0
        total = 0
        flag = True
        for images, labels in test_loader:
            # 获取对抗样本
            images = attack(images, labels)
            labels = labels.to(device)
            # 推理对抗样本
            outputs = model(images)
            # 获取预测值
            _, pre = torch.max(outputs.data, dim=1)
            # 计算对抗后的正确率
            total += batch_size
            correct += (pre == labels).sum()
            # 展示batch_size张图片 和 其预测的类别， flag -> 只打印第一个batch_size的图片
            if flag:
                # print(eps)
                savedir = r'C:\Users\99785\Desktop\\' + attack_method_str + '\\' + str(step) + '.png'
                show_image(torchvision.utils.make_grid(images.cpu().data, normalize=True), [attack_method_str + ' - step:', str(step) + '\n'] + [pre], title_flag=True, savedir=savedir)
                flag = False
        acc = 100 * float(correct) / total
        print('Accuracy of test text: %f %%' % acc, ' with using step: %f' % step)
        accuracy.append(acc)

    time.sleep(1)

    savedir = r'C:\Users\99785\Desktop\\' + attack_method_str + '\\statistic.png'
    plot_eps_acc(attack_method_str+' performance', steps, accuracy, savedir=savedir)


