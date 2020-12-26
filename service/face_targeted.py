#!/user/bin/env python    
#-*- coding:utf-8 -*-
import json
import pprint
import matplotlib.image

from torchvision.transforms import transforms
from torch.nn import functional as F
from Model.model_utils import set_seed
from service.attack_mapping import *
from utils.mysqldb import *



set_seed(6666)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_path = '../saveModel/myfacenet_1_1119.pkl'
model = torch.load(model_save_path)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # PIL Image → Tensor
])

json_data_path = r'../data/class_index.json'
data_mode = "face-12"
idx_to_class = json.load(open(json_data_path))[data_mode]

pp = pprint.PrettyPrinter(indent=2)


def predict(data, filename=None, save=False):
    # print(type(data))
    img_np = data.permute(1, 2, 0).cpu().numpy()
    # print(np.max(img_np))
    plt.imshow(img_np)
    # plt.show()
    if save:
        filename = './img/'+filename+'.png'
        print(np.max(img_np)) # [0,1]
        matplotlib.image.imsave(filename, img_np)
        insert_data(filename)
        os.remove(filename)

    data = torch.unsqueeze(data, 0).to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        # print('the predict class(idx): {}({})'.format(idx_to_class[str(pred.item())], pred.item()))

    predict_detail = {'class':'', 'index':'', 'probs':{}}
    print('The probs detail: ')
    for i, p in enumerate(idx_to_class):
        key = idx_to_class[str(i)].rjust(3)
        value = probs.cpu()[0][i].double().item()
        predict_detail['probs'][key] = round(value*100, 2)
        # print(idx_to_class[str(i)].rjust(3)+':','{:.6f}'.format(value))

    predict_class = idx_to_class[str(pred.item())]
    predict_detail['index'] = str(pred.item())
    predict_detail['class'] = predict_class
    pp.pprint(predict_detail)
    return predict_detail # 返回预测的概率


def generate(img, label, method, eps, mode='original'):
    img = torch.unsqueeze(img, 0).to(device)  # 输入到attack里要先增加一个维度
    img = attack_mapping(img, model, method=method, eps=eps, label=label, mode=mode)
    img = torch.squeeze(img)
    return img


if __name__ == '__main__':
    """注意定向攻击不能使用FGSM、FFGSM、StepLL"""

    """前端传来的图片  格式咋整？？"""
    # data
    path = r'D:\workspace\dataset\myfaces\train\DJX\1.png'
    img = Image.open(path)
    """先做转换"""
    data = transform(img).to(device) # 转换成torchtensor
    """定向这边不采取原图的class，而是前端传来的目标class"""
    pred_index = predict(data)['index']
    """前端传过来的eps"""
    eps = 0.6
    alpha = eps/40
    step = 20
    """原图预测类别 或者 前端传过来的攻击目标"""
    label = torch.tensor([10])
    """生成并预测对抗样本"""
    # predict_attack_img(img=data, label=label, method='APGD', eps=eps, alpha=alpha, step=step)

    """0 3 4 6 10"""
    # img_np = readData(1)
    # img_ts = transform(Image.fromarray(img_np)).to(device)
    # predict(img_ts)
