#!/user/bin/env python    
#-*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import pymysql as ml
import matplotlib as mpl

from PIL import Image
from matplotlib import pyplot as plt

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def insert_data(filename):
    fp = open(filename, 'rb')
    img_bytes = fp.read()
    fp.close()

    conn = ml.connect(host="121.37.5.209", user="root", password="0000", db="gan", port=3306)
    # 连接数据库对象

    cur = conn.cursor()
    # 游标对象

    sql = "insert into adv_img(adv_img, time) values(_binary %s, now())"
    # 定义好sql语句，%s是字符串的占位符

    try:
        cur.execute(sql, (img_bytes))
        # 执行sql语句
        conn.commit()
        # 提交到数据库中
    except Exception as e:
        # 捕获异常
        raise e
    finally:
        cur.close()
        conn.close()  # 关闭连接


def read_data(id=1):
    conn = ml.connect(host="121.37.5.209", user="root", password="0000", db="gan", port=3306)
    # 连接数据库对象

    cur = conn.cursor()
    # 游标对象

    sql = "select adv_img from adv_img where id = %s"
    # 定义好sql语句，%s是字符串的占位符

    try:
        cur.execute(sql, (id))
        # f = open('./img/image.png', 'wb')
        img_bytes = cur.fetchone()[0]
        img_np = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), 1)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        plt.imshow(img_np)
        # plt.show()
        # print(type(img_np), img_np.shape)
        # f.write(cur.fetchone()[0])
        # f.close()
        conn.commit()
        # 提交到数据库中
    except Exception as e:
        # 捕获异常
        raise e
    finally:
        cur.close()
        conn.close()  # 关闭连接

    return img_np


if __name__ == '__main__':

    # fp = open("./img/BIM.png", 'rb')
    #
    # img = fp.read()
    #
    # fp.close()

    # insertData(img)

    read_data(1)
    # 读取数据库
