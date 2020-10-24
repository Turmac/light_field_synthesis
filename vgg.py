"""
Implementation of VGG loss from "Photographic Image Synthesis with Cascaded Refinement Networks"

Date: 04/07/2019
Modified by: Qinbo Li
Modified based on pytorch/examples/fast_neural_style:
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
License:
https://github.com/pytorch/examples/blob/master/LICENSE
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

device = torch.device('cuda')


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        h = self.slice5(h)
        h_relu5_2 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_2, h_relu5_2)
        return out


def l1_loss(a, b):
    loss = torch.mean(torch.abs(a - b))
    return loss

def weighted_l1_loss(a, b, weight):
    loss = torch.abs(a - b)
    weighted_loss = torch.mean(weight*loss)
    return weighted_loss

def vgg_loss(x, y):
    """
    Args:
        x, y: input images, [B, C, H, W], after normalized by imagenet mean and std
    Reture:
        content loss
    """
    vgg = Vgg19(requires_grad=False).to(device)
    features_x = vgg(x)
    features_y = vgg(y)

    p1 = l1_loss(features_x.relu1_2, features_y.relu1_2)/2.6
    p2 = l1_loss(features_x.relu2_2, features_y.relu2_2)/4.8
    p3 = l1_loss(features_x.relu3_2, features_y.relu3_2)/3.7
    p4 = l1_loss(features_x.relu4_2, features_y.relu4_2)/5.6
    p5 = l1_loss(features_x.relu5_2, features_y.relu5_2)*10/1.5
    loss = p1 + p2 + p3 + p4 + p5

    return loss

def weighted_vgg_loss(x, y, weight):
    """
    Args:
        x, y: input images, [B, C, H, W], after normalized by imagenet mean and std
    Reture:
        weighted content loss
    """
    vgg = Vgg19(requires_grad=False).to(device)
    features_x = vgg(x)
    features_y = vgg(y)
    
    x1_2 = torch.unsqueeze(torch.sum(features_x.relu1_2, dim=1), dim=1)
    y1_2 = torch.unsqueeze(torch.sum(features_y.relu1_2, dim=1), dim=1)
    x2_2 = torch.unsqueeze(torch.sum(features_x.relu2_2, dim=1), dim=1)
    y2_2 = torch.unsqueeze(torch.sum(features_y.relu2_2, dim=1), dim=1)
    x2_2 = F.interpolate(x2_2, scale_factor=2.0, mode='bilinear')
    y2_2 = F.interpolate(y2_2, scale_factor=2.0, mode='bilinear')
    x3_2 = torch.unsqueeze(torch.sum(features_x.relu3_2, dim=1), dim=1)
    y3_2 = torch.unsqueeze(torch.sum(features_y.relu3_2, dim=1), dim=1)
    x3_2 = F.interpolate(x3_2, scale_factor=4.0, mode='bilinear')
    y3_2 = F.interpolate(y3_2, scale_factor=4.0, mode='bilinear')
    x4_2 = torch.unsqueeze(torch.sum(features_x.relu4_2, dim=1), dim=1)
    y4_2 = torch.unsqueeze(torch.sum(features_y.relu4_2, dim=1), dim=1)
    x4_2 = F.interpolate(x4_2, scale_factor=8.0, mode='bilinear')
    y4_2 = F.interpolate(y4_2, scale_factor=8.0, mode='bilinear')
    x5_2 = torch.unsqueeze(torch.sum(features_x.relu5_2, dim=1), dim=1)
    y5_2 = torch.unsqueeze(torch.sum(features_y.relu5_2, dim=1), dim=1)
    x5_2 = F.interpolate(x5_2, scale_factor=16.0, mode='bilinear')
    y5_2 = F.interpolate(y5_2, scale_factor=16.0, mode='bilinear')

    p1 = weighted_l1_loss(x1_2, y1_2, weight)
    p2 = weighted_l1_loss(x2_2, y2_2, weight)
    p3 = weighted_l1_loss(x3_2, y3_2, weight)
    p4 = weighted_l1_loss(x4_2, y4_2, weight)
    p5 = weighted_l1_loss(x5_2, y5_2, weight)
    loss = p1 + p2 + p3 + p4 + p5

    return loss
