# -*- coding: utf-8 -*-

import torch
import math
from jittor import nn
import torch.nn.functional as F


class MetaBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(MetaBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU() #(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU() #(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def execute(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MetaNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(MetaNetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def execute(self, x):
        return self.layer(x)


class MetaWideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(MetaWideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = MetaNetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = MetaNetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU() #(negative_slope=0.1, inplace=True)
        self.fc = MetaClassifier(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_gauss(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def execute(self, x, film_gamma=None, film_beta=None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(-1, self.channels)
        logit = self.fc(out)
        return logit


class MetaClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(MetaClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_gauss(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        logit = self.fc(x)
        return logit


class MetaNet(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(MetaNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            nn.ReLU(), #(inplace=True),
            nn.Linear(hid_dim, out_dim))

        for m in self.modules():
            # if isinstance(m, nn.Linear):
            #     # nn.init.xavier_gauss(m.weight.data)
            #     m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def execute(self, x_lab, x_unlab):
        logit_lab = self.fc(x_lab)
        logit_unlab = self.fc(x_unlab)
        return torch.cat([logit_lab, logit_unlab], dim=0)



