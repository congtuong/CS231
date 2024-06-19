from torch import nn
import torch
from torchvision.models import vgg16_bn
import numpy as np

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.bn_flag = bn
    def forward(self, x):
        x = self.conv(x)
        if self.bn_flag:
            x = self.bn(x)
        x = self.drop(self.relu(x))
        return x

class Yolov3VGG(nn.Module):
    def __init__(self,anchors, num_classes=2, vgg = vgg16_bn(pretrained=True),device='cuda'):
        super(Yolov3VGG, self).__init__()
        self.num_anchors = 5
        self.anchors = anchors
        self.num_classes = num_classes
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:-1])
        self.device = device
        # fit through the first 23 layers of vgg then add extra layers with 3 convs and a maxpool
        self.extra = nn.Sequential(
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # skip
        self.skip_module = nn.Sequential(
            CNNBlock(512, 64, kernel_size=1, padding=0, stride=1),
        )

        self.final = nn.Sequential(
            CNNBlock(256 + 512, 1024, kernel_size=3, padding=1,stride=1),
            CNNBlock(1024,256, kernel_size=3, padding=1,stride = 1),
            nn.Conv2d(256, self.num_anchors*(5 + self.num_classes),1),
        )
        self.init_values()

    def init_values(self):
        for c in self.final.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.extra.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.skip_module.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, x):
        # x is a batch of images
        x = torch.stack(x)
        # print(x.shape)
        output_size = x[0].shape[-1]
        output_size /= 32
        o_size = int(output_size)

        x = x.to(self.device)

        x = self.vgg_layers(x)

        skip_x = self.skip_module(x).to(self.device)

        skip_x = skip_x.view(-1, 64, o_size, 2, o_size, 2).contiguous()
        skip_x = skip_x.permute(0, 3, 5, 1, 2, 4).contiguous()
        skip_x = skip_x.view(-1, 256, o_size, o_size)

        x = self.extra(x)
        # print(x.shape)
        # print(skip_x.shape)
        x = torch.cat((x, skip_x), dim=1)
        x = self.final(x)
        return x