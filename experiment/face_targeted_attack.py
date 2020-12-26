#!/user/bin/env python    
#-*- coding:utf-8 -*- 

"""@@@
用于评估人脸身份识别对抗攻击算法的攻击效果

需要整合成可以迭代计算预先准备好的对抗算法的模式

"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from attack.attack_ffgsm import FFGSM
from attack.attack_fgsm import *
from utils.get_dataset import *
from utils.img_utils import plot_eps_acc
from Model.model_utils import set_seed


set_seed(6666)

"""获取数据"""
transform = transforms.Compose([
    # transforms.Scale(size),
    transforms.Resize((299, 299)),
    # transforms.CenterCrop((299, 299)),
    # transforms.RandomRotation(0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # PIL Image → Tensor
])

batch_size = 20
root = r'D:\workspace\dataset\myfaces\test'
_, test_loader = get_Dataloader('face-12', root=root, transform=transform, batch_size=batch_size, shuffle_flag=True)


"""获取预训练模型"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model = models.inception_v3(pretrained=True).to(device)
model_save_path = '../saveModel/myfacenet_1_1119.pkl'
model = torch.load(model_save_path).to(device)

# 直接切换到推理模式
model.eval()


"""未攻击正确率"""
print("True Image & Predicted Label")

correct = 0
total = len(test_loader.dataset)

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        # 前向传播
        outputs = model(images)
        # max 返回(最大值,下标)， pre size: [b]
        _, pre = torch.max(outputs.data, dim=1)
        # 累积正确个数
        correct += (pre == labels).sum()
        # 展示batch_size张图片 和 其预测的类别
        # showBatchImages(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre], False)

# 预测的平均正确率
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))




"""攻击后正确率"""
# 攻击参数
epss = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
# epss = [0.1,0.2,0.3] # 测试生成对抗样本的速度
# epss = np.linspace(0, 1.1, 100) #0.01
plot_eps = [0.1, 0.3, 0.5, 0.7, 0.9]

attack_methods = [FGSM, FFGSM]
attack_methods_str = ['FGSM', 'FFGSM']

# FGSM(Model).set_attack_mode(mode='targeted')

# attack_methods_with_steps = [BIM, StepLL, PGD] #, APGD, TPGD, CW, DeepFool]
# steps = [8, 8, 10]
# attack_methods_with_steps_str = ['BIM', 'StepLL', 'PGD']#, 'APGD', 'TPGD', 'CW', 'DeepFool']

loss = nn.CrossEntropyLoss()

print("Attack Image & Predicted Label")

# for attack_method, attack_method_str, step in zip(attack_methods_with_steps, attack_methods_with_steps_str, steps):
for attack_method, attack_method_str in zip(attack_methods, attack_methods_str):

    print(attack_method_str+' performance')
    accuracy = []
    for eps in epss:
        # attack = attack_method(Model, eps=eps, alpha=eps/66, steps=step)

        attack = attack_method(model, eps=eps)
        attack.set_attack_mode(mode='targeted')
        correct = 0
        total = len(test_loader.dataset)
        flag = True
        for images, labels in test_loader:
            # 获取对抗样本
            labels = torch.tensor([0]*labels.shape[0])
            labels = labels.to(device)
            images = attack(images, labels)

            # 推理对抗样本
            outputs = model(images)
            # 获取预测值
            _, pre = torch.max(outputs.data, dim=1)
            correct += (pre == labels).sum()
            # 展示batch_size张图片 和 其预测的类别
            if flag and eps in plot_eps:
                print(eps)
                show_image(torchvision.utils.make_grid(images.cpu().data, normalize=True), [attack_method_str + ' - eps:', str(eps) + '\n'] + [pre], True)
                flag = False
        acc = 100 * float(correct) / total
        print('Accuracy of test text: %f %%' % acc,' with using eps: %f' % eps)
        accuracy.append(acc)

    time.sleep(1)
    plot_eps_acc(attack_method_str+' performance', epss, accuracy)

