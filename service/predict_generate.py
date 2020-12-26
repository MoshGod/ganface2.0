#!/user/bin/env python    
#-*- coding:utf-8 -*-

import json
import pprint
from torchvision.transforms import transforms
from torch.nn import functional as F
from Model.model_utils import set_seed
from service.attack_mapping import *
from utils.mysqldb import *

set_seed(6666)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_save_path = '../saveModel/myfacenet_1_1119.pkl'
model_save_path = '../saveModel/resnet_18.pkl'
model = None

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # PIL Image → Tensor
])

json_data_path = r'../data/class_index.json'

data_mode = "imagenet"

# Model = models.resnet18(pretrained=True).to(device)
# torch.save(Model, model_save_path)

model = torch.load(model_save_path)
model = model.eval()
idx_to_class = json.load(open(json_data_path))[data_mode]
# print(idx_to_class)
# print(idx_to_class['775'])
pp = pprint.PrettyPrinter(indent=2)


def set_func(func: str):
    global data_mode, transform, idx_to_class, model_save_path, model
    if func in ['0', '1']:  # 人脸
        data_mode = 'face-12'
        model_save_path = '../saveModel/myfacenet_1_1119.pkl'
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # PIL Image → Tensor
        ])
    elif func == '2':  # 年龄
        data_mode = 'Asia-age-4'
        model_save_path = '../saveModel/Asiaage_resnet_4class.pkl'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # PIL Image → Tensor
        ])

    elif func == '3':  # 性别
        data_mode = 'gender'
        model_save_path = '../saveModel/mygendernet_0_1119.pkl'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
        ])

    elif func == '4':  # ImageNet-predict
        data_mode = 'imagenet'
        model_save_path = '../saveModel/resnet_18.pkl'
        transform = transforms.Compose([
            # transforms.Resize((299, 299)),
            transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    elif func == '5':  # ImageNet-generate
        data_mode = 'imagenet'
        model_save_path = '../saveModel/resnet_18.pkl'
        transform = transforms.Compose([
            transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
        ])

    idx_to_class = json.load(open(json_data_path))[data_mode]
    print(model)
    model = torch.load(model_save_path)
    model.eval()  # important


"""新增"""
def predict_imagenet(data):
    img_np = data.permute(1,2,0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()

    # 增加一维
    data = torch.unsqueeze(data, 0).to(device)

    # 关闭torch的梯度跟踪
    with torch.no_grad():
        logits = model(data)
        # print(type(logits), logits.shape)  # torch.Tensor  [1, 1000]
        probs = F.softmax(logits, dim=1)   # 按行作softmax
        # print(probs)
        pred = logits.argmax(dim=1)   # 返回logits最大的下标
        # print(pred)
        # print('the predict class(idx): {}({})'.format(idx_to_class[str(pred.item())], pred.item()))

    predict_detail = {'class': '', 'index': '', 'probs': {}}
    print('The probs detail: ')
    # print(pred.shape)
    # print(probs.topk(5, sorted=True))
    values, indices = probs.topk(5, sorted=True)  # 返回两组二维数据：values概率 + indices对应下标
    for value, indice in zip(values[0], indices[0]):
        key = idx_to_class[str(indice.cpu().item())][1]

        value = value.cpu().item()  # Tensor -> float

        predict_detail['probs'][key] = round(value*100, 2)
        # print(idx_to_class[str(i)].rjust(3)+':','{:.6f}'.format(value))

    predict_class = idx_to_class[str(pred.item())]
    predict_detail['index'] = str(pred.item())
    predict_detail['class'] = predict_class[1]
    pp.pprint(predict_detail)
    return predict_detail  # 返回预测概率详细信息


def predict(data, filename=None, save=False):
    # print(type(data))
    # img_np = data.permute(1,2,0).cpu().numpy()
    # print(np.max(img_np))
    # plt.imshow(img_np)
    # plt.show()
    # if save:
    #     filename = './img/'+filename+'.png'
    #     print(np.max(img_np)) # [0,1]
    #     matplotlib.image.imsave(filename, img_np)
    #     insertData(filename)
    #     os.remove(filename)

    data = torch.unsqueeze(data, 0).to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        # print('the predict class(idx): {}({})'.format(idx_to_class[str(pred.item())], pred.item()))

    predict_detail = {'class': '', 'index': '', 'probs': {}}
    print('The probs detail: ')
    for i, p in enumerate(idx_to_class):
        key = idx_to_class[str(i)]
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

    '''Main Function Test'''
    # # data
    # path = r'C:\Users\99785\Desktop\djx.png'
    # img = Image.open(path)
    # """先做转换"""
    # data = transform(img).to(device)  # 转换成torchtensor
    # """定向这边不采取原图的class，而是前端传来的目标class"""
    # # pred_index = predict(data)['index']
    # """前端传过来的eps"""
    # eps = 0.6
    # alpha = eps/40
    # step = 20
    # """原图预测类别 或者 前端传过来的攻击目标"""
    # # label = torch.tensor([10])
    # """生成并预测对抗样本"""
    # # predict_attack_img(img=data, label=label, method='APGD', eps=eps, alpha=alpha, step=step)
    # predict_imagenet(data)
    # """0 3 4 6 10"""
    # # img_np = readData(1)
    # # img_ts = transform(Image.fromarray(img_np)).to(device)
    # # predict(img_ts)

    '''ImageNet Test'''
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
    # print(data[0])
    ori_detail = predict_imagenet(data)
    # attacks[0](Model)
    data = transform2(const_image)
    # adv_img = generate(data, ori_detail['index'], 'DeepFool', 0.1)
    adv_img = generate(data, '789', 'FFGSM', 0.1, 'targeted')
    # adv_img = generate(data, ori_detail['index'], 'MIFGSM', 0.8)
    adv_detail = predict_imagenet(adv_img)

    '''Targeted Test'''
    # img_path = r'C:\Users\99785\Desktop\86.png'
    # image = Image.open(img_path)
    # set_func('1')
    # data = transform(image)
    #
    # ori_detail = predict(data)
    # adv_img = generate(data, '1', 'APGD', 1, 'targeted')
    # adv_detail = predict(adv_img)





