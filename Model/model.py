#!/user/bin/env python
#-*- coding:utf-8 -*-

import math
from Model.model_utils import *
from torch import nn
from torch.nn import functional as F


# 默认参数，不是很重要，但可以学习
from Model.model_utils import Flatten

DIM_INPUT = 15
NUM_CLASS = 10
BATCH_SIZE = 16

IMAGE_SIZE = 16
COLOR_CHANNEL = 3


class SimpleModel(nn.Module):
    def __init__(self, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleImageModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASS):
        super(SimpleImageModel, self).__init__()
        self.num_classes = NUM_CLASS
        self.conv1 = nn.Conv2d(
            COLOR_CHANNEL, 8, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(4)
        self.linear1 = nn.Linear(4 * 4 * 8, self.num_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class MLP(nn.Module):
    # MLP-300-100

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 300)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


class ConvModel(nn.Module):

    def __init__(self, image_size=IMAGE_SIZE, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(ConvModel, self).__init__()
        self.stride = 2
        if image_size == 28:
            self.fl_size = 64
            self.stride = 1
        elif image_size == 128:
            self.fl_size = 3136
        elif image_size == 299:
            self.fl_size = 18496
        elif image_size == 224:
            self.fl_size = 10816
        else:
            self.fl_size = 1

        self.conv_unit = nn.Sequential(
            nn.Conv2d(dim_input, 16, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(self.fl_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # print(x.shape)
        x = x.view(batchsz, self.fl_size)
        logits = self.fc_unit(x)

        return logits


class ConvWidthModel(nn.Module):
    def __init__(self):
        super(ConvWidthModel, self).__init__()

        self.conv_unit1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Flatten()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv_unit1(x)
        # print(x.size())
        x = self.fc1(x)
        logits = self.fc2(x)

        return logits


class LeNet5Madry(nn.Module):

    def __init__(
            self, nb_filters=(1, 32, 64), kernel_sizes=(5, 5),
            paddings=(2, 2), strides=(1, 1), pool_sizes=(2, 2),
            nb_hiddens=(7 * 7 * 64, 1024), nb_classes=10):
        super(LeNet5Madry, self).__init__()
        self.conv1 = nn.Conv2d(
            nb_filters[0], nb_filters[1], kernel_size=kernel_sizes[0],
            padding=paddings[0], stride=strides[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(pool_sizes[0])
        self.conv2 = nn.Conv2d(
            nb_filters[1], nb_filters[2], kernel_size=kernel_sizes[1],
            padding=paddings[0], stride=strides[0])
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(pool_sizes[1])
        self.linear1 = nn.Linear(nb_hiddens[0], nb_hiddens[1])
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(nb_hiddens[1], nb_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out




if __name__ == '__main__':
    x = torch.rand((1,1,28,28))

    # trained_model = resnet18(pretrained=True)  # 设置True，表明使用训练好的参数  224*224
    # Model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
    #                       Flatten(),  # [b, 512, 1, 1] => [b, 512]
    #                       # nn.Linear(512, 5)
    #                       )

    # x = batch_per_image_standardization(x)
    # Model = nn.Sequential(
    #     PerImageStandardize(),
    #     WideResNet(124, 10, 1)
    # )
    model = ConvModel(image_size=28, dim_input=1, num_classes=10) # 年龄
    # ConvModel(image_size=299, num_classes=12) # 人脸
    # ConvModel(image_size=224, num_classes=2)  # 性别
    print(model(x).size())
