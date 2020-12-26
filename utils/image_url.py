#!/user/bin/env python
#-*- coding:utf-8 -*-


import sys
import oss2
import uuid
import os
import urllib.request

from torchvision import utils as vutils
from matplotlib import pyplot as plt
from PIL import Image


sys.path.append("..")
'''
https://blog.csdn.net/weixin_42812527/article/details/81638604
https://blog.csdn.net/weixin_40475396/article/details/80584461
https://blog.csdn.net/LoganPython/article/details/81176825?utm_source=blogxgwz5
'''
# from oss2.config import ALIYUN_OSS_SETTING


class AliyunOss(object):

    def __init__(self):
        self.access_key_id = "LTAI4G3Zm7rWz6gTHtAWf1yj"
        self.access_key_secret = "G8EZDb07oFWWMeoU1WBWHT8oxmeiLM"
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket_name = "ganface1"
        self.endpoint = "oss-cn-beijing.aliyuncs.com"
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    def put_file(self, name, file):
        """
        :param name: 文件名
        :param file: 文件
        :return:
        """
        # Endpoint以杭州为例，其它Region请按实际情况填写。
        result = self.bucket.put_object(name, file)
        # HTTP返回码。
        print('http status: {0}'.format(result.status))
        # 请求ID。请求ID是请求的唯一标识，强烈建议在程序日志中添加此参数。
        print('request_id: {0}'.format(result.request_id))
        # ETag是put_object方法返回值特有的属性。
        print('ETag: {0}'.format(result.etag))
        # HTTP响应头部。
        print('date: {0}'.format(result.headers['date']))

    def put_object_from_file(self, name, file):
        """
        上传本地文件
        :param name: 需要上传的文件名
        :param file: 本地文件名
        :return: 阿里云文件地址
        """
        # 调用bucket类函数
        self.bucket.put_object_from_file(name, file)
        return "https://{}.{}/{}".format(self.bucket_name, self.endpoint, name)

    def put_object(self, name, file):
        # 上传二进制文件
        self.bucket.put_object(name, file)
        return "https://{}.{}/{}".format(self.bucket_name, self.endpoint, name)


# 上传图片至服务器并返回图片url
def upload(img) -> str:
    local_path = r'../data/temp.jpg'
    vutils.save_image(img, local_path)
    aliyunoss = AliyunOss()
    # uuid: 通用唯一标识符
    img_key = str(uuid.uuid4()) + '.jpg'
    adv_img_url = aliyunoss.put_object_from_file('advimg/' + img_key, local_path)
    os.remove(local_path)
    return adv_img_url

def upload_1():
    import oss2
    import datetime
    import string
    import random
    import requests

    # 自定义随机名称
    now = datetime.datetime.now()
    random_name = now.strftime("%Y%m%d%H%M%S") + ''.join([random.choice(string.digits) for _ in range(4)])
    # 自有域名
    cname = 'https://****.com/'
    # 存放OSS路径
    file_name = '****/{}.jpg'.format(random_name)
    # AccessKeyID和AccessKeySecret
    auth = oss2.Auth('****', '****')
    # 外网访问的Bucket域名和Bucket名称
    bucket = oss2.Bucket(auth, '*****', '*****', is_cname=True)
    # 图片链接
    url = '********'
    resp = requests.get(url).content
    bucket.put_object(file_name, resp)
    # 最终的图片链接
    get_url = cname + file_name
    print(get_url)

def download_from_url():

    image_url = 'http://img.jingtuitui.com/759fa20190115144450401.jpg'
    #
    file_path = r'C:\Users\99785\Desktop\mytest.jpg'

    # file_name = image_url

    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)  # 如果没有这个path则直接创建
        file_suffix = os.path.splitext(image_url)[1]
        print(file_suffix)
        filename = '{}{}'.format(file_path, file_suffix)
        # 拼接文件名。
        print(filename)
        urllib.request.urlretrieve(image_url, filename=filename) #利用urllib.request.urltrieve方法下载图片
        print('Download success.')

    except IOError as e:
        print(1, e)

    except Exception as e:
        print(2, e)


def main():
    aliyunoss = AliyunOss()
    # # img = aliyunoss.put_object("传到阿里云上的图片名", "二进制图片")
    img_url = aliyunoss.put_object_from_file("dog.jpg", r"C:\Users\99785\Desktop\dog.jpg")
    # # print(type(img_url), img_url)
    # download_from_url()
    print(img_url)
    # image_url = 'https://ganface.oss-cn-shenzhen.aliyuncs.com/age/test.jpg'
    # urllib.request.urlretrieve(image_url, filename=r'C:\Users\99785\Desktop\mytest.jpg')

    from skimage import io
    # image = io.imread(image_url)  # numpy.ndarray格式

    # # 转化为PIL格式
    # image = Image.fromarray(image)
    #
    # # print(type(image))
    # # io.imshow(image)
    # # io.show()
    # plt.imshow(image)
    # plt.show()

if __name__ == '__main__':
    main()