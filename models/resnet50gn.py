import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

class GroupNormConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(groups, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class ResNetStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GroupNormConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return self.pool(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, groups=32, base_width=64, downsample=None):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(groups, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

def make_layer(in_planes, planes, blocks, stride=1, groups=32, base_width=64):
    downsample = None
    if stride != 1 or in_planes != planes * 4:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(groups, planes * 4),
        )

    layers = [ResNetBlock(in_planes, planes, stride, groups, base_width, downsample)]
    for _ in range(1, blocks):
        layers.append(ResNetBlock(planes * 4, planes, 1, groups, base_width))
    return nn.Sequential(*layers)

class ResNet50GN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = ResNetStem()
        self.layer1 = make_layer(64, 64, 3)
        self.layer2 = make_layer(256, 128, 4, stride=2)
        self.layer3 = make_layer(512, 256, 6, stride=2)
        self.layer4 = make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # res2
        x = self.layer2(x)  # res3
        x = self.layer3(x)  # res4
        x = self.layer4(x)  # res5
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def build_resnet50_gn_from_detectron2(num_classes=1000):
    return ResNet50GN(num_classes)
