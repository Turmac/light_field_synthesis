import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma
import torch.optim as optim
from PIL import Image
import skimage
import matplotlib.pyplot as plt
from os import path


device = torch.device('cuda')
lfsize = [372, 540, 8, 8]
lfsize_train = [368, 528, 8, 8]
lfsize_test = [368, 528, 8, 8]

def denormalize_lf(lf):
    return lf/2.0+0.5

class LightFieldNpDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        self.path = path
        self.mode = mode
        self.filenames = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lf = np.load(self.filenames[idx])
        lf = np.squeeze(lf, axis=0)
        lf = torch.tensor(lf, dtype=torch.float)

        conner_1 = lf[:, :, 0, 0, :]
        conner_2 = lf[:, :, 0, 7, :]
        conner_3 = lf[:, :, 7, 0, :]
        conner_4 = lf[:, :, 7, 7, :]
        aif = lf[:,:,4,4,:]

        return (self.filenames[idx], [conner_1, conner_2, conner_3, conner_4, aif])

class LightFieldDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        self.path = path
        self.mode = mode
        self.filenames = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lf = Image.open(self.filenames[idx])
        
        lf = np.asarray(lf)
        lf = lf[:372*14,:540*14,:3]
        lf = np.reshape(lf, [372, 14, 540, 14, 3])
        lf = lf[:, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2), :, (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2), :]
        lf = np.transpose(lf, (0, 2, 1, 3, 4))

        corner1 = skimage.exposure.adjust_gamma(lf[:, :, 0, 0, :], 0.4)
        corner2 = skimage.exposure.adjust_gamma(lf[:, :, 0, 7, :], 0.4)
        corner3 = skimage.exposure.adjust_gamma(lf[:, :, 7, 0, :], 0.4)
        corner4 = skimage.exposure.adjust_gamma(lf[:, :, 7, 7, :], 0.4)
        aif = skimage.exposure.adjust_gamma(lf[:, :, 4, 4, :], 0.4)

        return (self.filenames[idx], [corner1, corner2, corner3, corner4, aif])
    
    def get_filename(self, idx):
        return self.filenames[idx]

if __name__ == '__main__':
    dataset = LightFieldDataset('F:/Datasets/TAMULF2')
    dst_path = 'E:/data/TAMULF_4corner_img/'
    cnt = 0

    for i in range(len(dataset)):
        cnt += 1
        
        filename = dataset.get_filename(i)
        filename = filename.split('\\')[-1]
        filename = filename.split('.')[0]

        # check if file exists
        dst_filename = '%s_%d.png' % (filename, 0)
        if path.exists(dst_path+dst_filename):
            print('skip %d'%cnt, end='\r')
            continue

        data = dataset[i]
        for i in range(4):
            dst_filename = '%s_%d.png' % (filename, i)
            #plt.imsave(dst_path+dst_filename, denormalize_lf(data[1][i]))
            plt.imsave(dst_path+dst_filename, data[1][i])

        # save the center
        #dst_filename = '%s_center.png' % (filename)
        #plt.imsave(dst_path+dst_filename, data[1][4])
        print(filename, cnt, end='\r')

    '''
    for data in dataset:
        cnt += 1

        filename = data[0].split('\\')[-1]
        filename = filename.split('.')[0]

        # check if file exists
        dst_filename = '%s_%d.png' % (filename, 0)
        if path.exists(dst_path+dst_filename):
            print('skip %d'%cnt)
            continue

        for i in range(4):
            dst_filename = '%s_%d.png' % (filename, i)
            #plt.imsave(dst_path+dst_filename, denormalize_lf(data[1][i]))
            plt.imsave(dst_path+dst_filename, data[1][i])

        # save the center
        #dst_filename = '%s_center.png' % (filename)
        #plt.imsave(dst_path+dst_filename, data[1][4])
        print(filename, cnt, end='\r')
    '''