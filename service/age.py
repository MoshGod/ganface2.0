#!/user/bin/env python
#-*- coding:utf-8 -*-
import json
import os
import pprint
import matplotlib
import numpy as np
import torch

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torch.nn import functional as F
from Model.model_utils import set_seed
from utils.mysqldb import insert_data

from service.attack_mapping import *

set_seed(6666)

# 选择设备，加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = '../saveModel/Asiaage_resnet_4class.pkl'
model = torch.load(model_save_path)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # PIL Image → Tensor
])

json_data_path = r'../data/class_index.json'
data_mode = "Asia-age-4"
# a dict
idx_to_class = json.load(open(json_data_path))[data_mode]


# 美观输出python对象, 缩进两个字符
pp = pprint.PrettyPrinter(indent=2)


def predict(data, filename=None, save=False):
    img_np = data.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    # plt.show()
    if save:
        filename = './img/'+filename+'.png'
        print(np.max(img_np)) # [0,1]
        matplotlib.image.imsave(filename, img_np)
        insert_data(filename)
        os.remove(filename)
    # 在第0维增加一个维度
    data = torch.unsqueeze(data, 0).to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        # print('the predict class(idx): {}({})'.format(idx_to_class[str(pred.item())], pred.item()))
        # print(type(pred.item()))

    predict_detail = {'class':'', 'index':'', 'probs':{}}
    print('The probs detail: ')
    for i, p in enumerate(idx_to_class):
        key = idx_to_class[str(i)].rjust(3)
        value = probs.cpu()[0][i].double().item()
        predict_detail['probs'][key] = round(value*100, 2)
        # print(idx_to_class[str(i)].rjust(3)+':','{:.6f}'.format(value))
    # print(type(pred), pred)

    predict_class = idx_to_class[str(pred.item())]
    predict_detail['index'] = str(pred.item())
    predict_detail['class'] = predict_class
    pp.pprint(predict_detail)
    return predict_detail  # 返回预测的概率和 对抗样本


def generate(img, label, method, eps, mode='original'):
    img = torch.unsqueeze(img, 0).to(device)  # 输入到attack里要先增加一个维度
    img = attack_mapping(img, model, method=method, eps=eps, label=label, mode=mode)
    img = torch.squeeze(img)
    return img


if __name__ == '__main__':
    """前端传来的图片  格式咋整？？"""
    # data
    # path = r'D:\WorkSpace\Python\Introduction_to_Software_Engineering\dataset\face\test\QC\48.jpg'
    path = r'C:\Users\99785\Desktop\djx.png'
    img = Image.open(path)
    # print(type(img))

    # fetch image from url
    from skimage import io
    image_url = 'https://ganface.oss-cn-shenzhen.aliyuncs.com/age/test.jpg'
    image = io.imread(image_url)
    image = Image.fromarray(image)  # 转化为PIL格式

    """先做转换"""
    data = transform(img).to(device) # 转换成torchtensor 和 size
    """预测原图"""
    pred_index = predict(data)['index']
    # print(type(pred_index), pred_index)
    """前端传过来的eps"""
    eps = 0.05
    """预测对抗样本"""
    # adversarial_img, send_msag = generate_attack_img(img=data, method='APGD', eps=eps, alpha=eps / 66, label=pred_index, step=4)
    # print(type(transforms.ToPILImage(adversarial_img)))

    # 封装为json格式
    # send_msag['index'] = str(send_msag['index'].item())
    # send_msag['img_url'] = image_url
    # json_str = json.dumps(send_msag)
    # pp.pprint(send_msag)

    # adversarial_img.resize_(3, 271, 177)
    # img_np = adversarial_img.permute(1, 2, 0).cpu().numpy()
    # plt.imshow(img_np)
    # plt.show()


    # 保存到本地
    # from torchvision import utils as vutils
    # vutils.save_image(adversarial_img, r'C:\Users\99785\Desktop\test.jpg')
    # 直接保存至oss
    # from utils.image_url import AliyunOss
    #
    # aliyunoss = AliyunOss()
    #
    # img_url = aliyunoss.put_object_from_file("age/test1.jpg", adversarial_img)
    # print(type(img_url), img_url)

    # print(type(adversarial_img), adversarial_img.shape)


    # 发送requests请求
    # import requests
    # url = ''
    # requests.post(url, json=json_str)