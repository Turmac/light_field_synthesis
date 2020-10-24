"""
Use a deep network to infer color transfer parameters
"""
import os
import random
import time
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
from torchvision import models
import torch.optim as optim
from PIL import Image
import skimage
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from vgg import vgg_loss, weighted_vgg_loss


device = torch.device('cuda')
batch_size = 1
train_batch_size = 8
lfsize = [372, 540, 8, 8]
lfsize_train = [368, 528, 8, 8]
lfsize_test = [368, 528, 8, 8]
mode = 'train'
resume = False

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def denormalize(lf):
    return lf/2.0+0.5

class ColorNet(nn.Module):
    def __init__(self, requires_grad=False):
        super(ColorNet, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        self.vgg5_4 = torch.nn.Sequential()
        for x in range(36):
            self.vgg5_4.add_module(str(x), vgg_pretrained_features[x])
        
         # fc layers
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3*256)
        )
        
        if not requires_grad:
            for name, param in self.named_parameters():
                if 'vgg' in name:
                    param.requires_grad = False
    
    def forward(self, X):
        features = self.vgg5_4(X)
        
        x = self.avgpool(features.detach())
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.tanh(x)

        return x

def l1_loss(x, y):
    loss = torch.mean(torch.abs(x - y))
    return loss

def color_transform(src, T):
    """
    src: [H, W, 3]
    T: [3, 256]
    """
    res = torch.tensor(src, dtype=torch.float32)
    #res = np.zeros_like(src, dtype=np.float)

    for i in range(3):
        for j in range(src.shape[0]):
            for k in range(src.shape[1]):
                loc = src[j,k,i].item()
                res[j][k][i] = T[i, loc]*1.0
                if res[j,k,i] < 0:
                    res[j,k,i] = 0
                if res[j,k,i] > 1:
                    res[j,k,i] = 1
    return res

def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)

model = ColorNet()
model = model.to(device=device)

params_to_update = model.parameters()
optimizer = optim.Adam(params_to_update, lr=0.0002)



def train(model):
    model.train()  # Set model to training mode

    for i in range(10000):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            src = torch.tensor(np.array(Image.open('E:/data/tmp/0.jpg')))
            tgt = torch.tensor(np.array(Image.open('E:/data/tmp/1.jpg')))
            src_norm = torch.tensor(src, dtype=torch.float32)/255.0
            tgt_norm = torch.tensor(tgt, dtype=torch.float32)/255.0
            inputs = torch.unsqueeze(src_norm.permute([2,0,1]), dim=0)
            x = torch.tensor(inputs, dtype=torch.float32).to(device)

            y = model(x)
            T = torch.reshape(y, (3, 256))
            T = denormalize(T)
            print(T[:,:10])

            v = random.randint(0, 69)
            v = 0

            src = src[v:v+10,:,:]
            tgt_norm = tgt_norm[v:v+10,:,:]

            output = color_transform(src, T)

            # loss
            loss = l1_loss(output, tgt_norm)
            loss.backward()
            optimizer.step()

        print('loss of iter %d: %lf'%(i, loss.item()), end='\r')

        if (i+1)%100 == 0:
            # save model
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, filename='checkpoints/ct_%d.pth.tar'%(i+1))
            np.save('ct.npy', T.detach().cpu().numpy())


def main():
    train(model)

if __name__ == '__main__':
    main()
    print('finished')
