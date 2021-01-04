#!/user/bin/env python
#-*- coding:utf-8 -*-
import sys
import time

import torch

class Attack(object):
    r"""
    攻击方法的父类：
        自动将device设置为给定型号所在的device；
        由于是攻击已训练好的模型，它将模型的训练模式临时更改为 eval，forward完会改完去。
    """

    def __init__(self, name, model):
        r"""
        初始化内部攻击状态。

        参数:
            name (str) : 算法名
            model (torch.nn.Module): 被攻击的模型
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0] # 获取模型名字

        self.training = model.training # 模型初始模式
        self.device = next(model.parameters()).device

        self._targeted = 1 # 非定向攻击
        self._attack_mode = 'original'
        self._return_type = 'float'

    def forward(self, *input, **kwargs):
        r"""
        定义了每次调用__call__时执行的计算，被所有子类重写。
        """
        raise NotImplementedError

    def set_attack_mode(self, mode):
        r"""
        设置攻击模式。

        参数:
            mode (str) : 'original' 默认。
                         'targeted' 使用输入标签作为目标标签。
                         'least_likely' 使用最不可能的标签作为目标标签。
        """
        if mode == "original":
            self._attack_mode = "original"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(mode + " is not a valid mode. [Options : original, targeted, least_likely]")

    def set_return_type(self, type):
        r"""
        设置对抗图像的返回类型: int or float.

        参数:
            type (str) : float or int. (DEFAULT : 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, data_loader, save_path=None, verbose=True):
        r"""
        从给定的torch.utils.data.DataLoader将对抗图像另存为torch.tensor。

        参数:
            data_loader (torch.utils.data.DataLoader) : 数据加载器
            save_path (str) : 保存路径
            verbose (bool) : 是否显示详细信息。 (DEFAULT : True)
        使用：
            attack.save(data_loader=test_loader, save_path="./data/temp.pt", verbose=True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            # 调用forward生成对抗图像
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float() / 255

            if verbose: # 打印进度
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print("- Save Progress : %2.2f %% / Accuracy : %2.2f %%" % ((step + 1) / total_batch * 100, acc))

        # 按第0维拼接回dataloader的形状
        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)

        if save_path is not None:
            torch.save((x, y), save_path)
            print('\n- Save Complete!')
        # 切换为原模式
        self._switch_model()

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels

    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels

    def _get_least_likely_label(self, images, labels):
        r"""
        返回the least likely label
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        r"""
        把image里值的类型改为uint
        """
        return (images * 255).type(torch.uint8)

    def _switch_model(self):
        r"""
        用于还原模型初始的训练模式。
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        r"""
        打印对象信息
        """
        # 获取对象属性，为字典类型
        info = self.__dict__.copy()
        # print(info)

        del_keys = ['model'] # 不显示的model属性

        for key in info.keys(): # 不显示私有属性('_'开头)
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys: # 删除info中不显示的属性
            del info[key]

        info['attack_mode'] = self._attack_mode
        # if info['attack_mode'] == 'only_original':
        #     info['attack_mode'] = 'original'

        info['return_type'] = self._return_type
        # 算法名()
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images



if __name__ == '__main__':
    a = Attack('FGSM', torch.nn.Sequential(torch.nn.Linear(10,1)).to('cpu'))
    print(a)
    print()
    print(torch.nn.Sequential(torch.nn.Linear(10,1)))
