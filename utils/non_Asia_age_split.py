#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import os
import random
import shutil

global_id = 0


def copy_file(file_dir, tar_dir):
    global global_id
    pathDir = os.listdir(file_dir)

    for name in pathDir:
        shutil.copyfile(file_dir + '\\' + name, tar_dir + '\\tmp' + str(global_id) + name)
        global_id += 1


if __name__ == '__main__':
    fileDir = r'D:\workspace\dataset\AGFW_cropped\cropped\128\male\age_'
    tarDir = r'D:\workspace\dataset\non_Asia_age_11\train\age_'

    dir_list1 = ['10_14', '15_19', '20_24', '25_29']
    dir_list2 = ['30_34', '35_39', '40_44', '45_49']
    dir_list3 = ['50_54', '55_59', '60_94']
    # dir_list = [dir_list1, dir_list2, dir_list3]
    dir_list = dir_list1 + dir_list2 + dir_list3
    # dir_dst = ['10-29', '30-49', '50-94']
    for i, dst in enumerate(dir_list):
        # for y in dir_list:
        #     print(y)
            copy_file(fileDir + dst, tarDir + dst)

