# -*- coding: utf-8 -*-

from service import predict_generate
import json
from flask import Flask, make_response, request
from skimage import io
from utils.image_url import *
from PIL import Image
from Model.model_utils import set_seed

set_seed(6666)

app = Flask(__name__)


@app.route("/pre")
def predict_image():
    func = str(request.args["func"])  # 0, 1, 2, 3, 4, 5
    image_url = str(request.args["image_url"])
    result_data = {}
    img = io.imread(image_url)
    img = Image.fromarray(img).convert("RGB")  # 转化为PIL格式，并兼容png格式

    # if func in ['0', '1']: # 人脸
    #     pre_adv.set_func('0')
    #     data = pre_adv.transform(img)
    #     result_data = pre_adv.predict(data=data)
    # elif func == '2': # 年龄
    #     pre_adv.set_func('2')
    #     data = pre_adv.transform(img)
    #     result_data = pre_adv.predict(data=data)
    # elif func == '3': # 性别
    #     pre_adv.set_func('3')
    #     data = pre_adv.transform(img)
    #     result_data = pre_adv.predict(data=data)
    # else   # ImageNet
    #     pre_adv.set_func('4')
    #     data = pre_adv.transform(img)
    #     result_data = pre_adv.predict(data=data)

    predict_generate.set_func(func)
    data = predict_generate.transform(img)
    if func == '4' or func == '5':
        result_data = predict_generate.predict_imagenet(data=data)
    else:
        result_data = predict_generate.predict(data=data)

    # print(result_data)

    rst = make_response(json.dumps(result_data))
    rst.headers['Access-Control-Allow-Origin'] = '*'
    return rst


@app.route("/adv")
def generate_adversarial_image():

    ###  特别注意：当选择Imagenet的Generate功能时， func设为5
    func = str(request.args["func"])  # 0,1,2,3, 5
    attack_method = str(request.args["attack_method"])  # 0-7
    level = float(request.args["level"])  # 攻击强度
    image_url = str(request.args["image_url"])
    targeted_index = str(request.args["targeted_index"])  # 若非定向则不用传此参数

    ori_pre_details = {}
    adv_pre_details = {}

    img = io.imread(image_url) # ndarray
    img = Image.fromarray(img).convert("RGB")   # 转化为PIL格式，并兼容png格式

    # 映射算法  之后可以把json映射封装成函数   不需要映射

    # json_data_path = r'../data/class_index.json'
    # data_mode = "Algorithm"
    # attack_method = json.load(open(json_data_path))[data_mode][method_idx]

    # if func == '0':  # 非定向
    #     pre_adv.set_func('0')
    #     data = pre_adv.transform(img)
    #     ori_pre_details = pre_adv.predict(data)
    #     pred_index = ori_pre_details['index']
    #     adv_img = pre_adv.generate(img=data, method=attack_method, eps=level, alpha=level / 66,
    #                                   label=pred_index, step=4)
    #     adv_pre_details = pre_adv.predict(adv_img)
    # elif func == '1':  # 定向
    #     pre_adv.set_func('1')
    #     data = pre_adv.transform(img)
    #     ori_pre_details = pre_adv.predict(data)
    #     adv_img = pre_adv.generate(img=data, method=attack_method, eps=level, alpha=level / 66,
    #                                   label=targeted_index, step=4)
    #     adv_pre_details = pre_adv.predict(adv_img)
    # elif func == '2':  # 年龄
    #     pre_adv.set_func('2')
    #     data = pre_adv.transform(img)
    #     ori_pre_details = pre_adv.predict(data)
    #     pred_index = ori_pre_details['index']
    #     adv_img = pre_adv.generate(img=data, method=attack_method, eps=level, alpha=level / 66,
    #                                   label=pred_index, step=4)
    #     adv_pre_details = pre_adv.predict(adv_img)
    # elif func == '3':  # 性别
    #     pre_adv.set_func('3')
    #     data = pre_adv.transform(img)
    #     ori_pre_details = pre_adv.predict(data)
    #     pred_index = ori_pre_details['index']
    #     adv_img = pre_adv.generate(img=data, method=attack_method, eps=level, alpha=level / 66,
    #                                   label=pred_index, step=4)
    #     adv_pre_details = pre_adv.predict(adv_img)
    # else:  # ImageNet
    #     pre_adv.set_func('4')
    #     data = pre_adv.transform(img)
    #     ori_pre_details = pre_adv.predict(data)
    #     pred_index = ori_pre_details['index']
    #     adv_img = pre_adv.generate(img=data, method=attack_method, eps=level, alpha=level / 66,
    #                                label=pred_index, step=4)
    #     adv_pre_details = pre_adv.predict(adv_img)

    predict_generate.set_func(func)
    data = predict_generate.transform(img)
    if func == '5':
        predict_generate.set_func('4')
        data = predict_generate.transform(img)
        ori_pre_details = predict_generate.predict_imagenet(data)
        predict_generate.set_func('5')
        data = predict_generate.transform(img)
    else:
        ori_pre_details = predict_generate.predict(data)

    pred_index = ori_pre_details['index']

    if targeted_index >= '0':    # 定向攻击指定攻击编号
        pred_index = targeted_index
        adv_img = predict_generate.generate(img=data, method=attack_method, eps=level,
                                            label=pred_index, mode='targeted')
    else:
        adv_img = predict_generate.generate(img=data, method=attack_method, eps=level,
                                            label=pred_index)
    if func == '5':
        adv_pre_details = predict_generate.predict_imagenet(adv_img)
    else:
        adv_pre_details = predict_generate.predict(adv_img)

    adv_img_url = upload(adv_img)  # 图片保存到本地上传至服务器, 返回url


    # 返回index
    return_data = {
        'adv_img_url': adv_img_url,
        'origin_index': ori_pre_details['index'],
        'origin_class': ori_pre_details['class'],
        'origin_conf': ori_pre_details['probs'][ori_pre_details['class']],
        'adv_index': adv_pre_details['index'],
        'adv_class': adv_pre_details['class'],
        'adv_conf': adv_pre_details['probs'][adv_pre_details['class']]
    }
    rst = make_response(json.dumps(return_data))
    rst.headers['Access-Control-Allow-Origin'] = '*'
    return rst


if __name__ == '__main__':
    app.run()


'''
test
预测：http://127.0.0.1:5000/pre?func=2&image_url=https://ganface.oss-cn-shenzhen.aliyuncs.com/age/test.jpg
生成：http://127.0.0.1:5000/adv?func=2&attack_method=APGD&level=0.1&image_url=https://ganface.oss-cn-shenzhen.aliyuncs.com/age/test.jpg&targeted_index=0

Imagenet: http://127.0.0.1:5000/adv?func=5&attack_method=CW&level=0.1&image_url=https://ganface.oss-cn-shenzhen.aliyuncs.com/dog.jpg
对抗图片：https://ganface.oss-cn-shenzhen.aliyuncs.com/age/d2778f46-2ad1-429d-bef3-121fd49a184f.jpg 
'''