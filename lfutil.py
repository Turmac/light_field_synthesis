import os
from os import listdir
from os.path import isfile, join
import numpy as np
import skimage
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt

lfsize = [372, 540, 8, 8]

def normalize_lf(lf):
    return 2.0*(lf-0.5)

def denormalize_lf(lf):
    return lf/2.0+0.5

def read_eslf(filename):
    lf = io.imread(filename)
    lf = skimage.exposure.adjust_gamma(lf, 0.4)

    lf = np.array(lf)
    lf = lf[:372*14, :540*14, :3]/255.0

    lf = np.reshape(lf, (372, 14, 540, 14, 3))
    lf = np.transpose(lf, (0, 2, 1, 3, 4))
    lf = lf[:, :, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2), (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2), :]

    return lf


def display_lf(lf, lfsize):
    box = 10
    box_w = lfsize[2]*10
    
    img = None
    for i in range(lfsize[2]):
        for j in range(lfsize[3]):
            
            j2 = j if i%2 == 0 else lfsize[2]-1-j

            frame = lf[:,:,i,j2,:]
            frame = np.clip(frame, 0, 1)

            x = i*box
            y = j2*box
            frame[0:10, 0:10] = 1
            frame[0:10, box_w-10:box_w] = 1
            frame[box_w-10:box_w, 0:10] = 1
            frame[box_w-10:box_w, box_w-10:box_w] = 1
            frame[x:x+box, y:y+box] = 0.5

            if img is None:
                img = plt.imshow(frame)
            else:
                img.set_data(frame)
                plt.pause(0.01)
                plt.draw()

def display_lf_folder(path):
    files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    files = files[600:]
    
    for f in files:
        print(f)
        lf = read_eslf(f)
        display_lf(lf, lfsize)

def display_aif_lf_folder(path):
    files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    files = files[200:]

    img = None
    for f in files:
        lf = read_eslf(f)
        aif = lf[:, :, lfsize[2]//2, lfsize[3]//2, :]
        if img is None:
            img = plt.imshow(aif)
        else:
            img.set_data(aif)
            plt.pause(0.01)
            plt.draw()

def display_4corner(lf):
    corners = (lf[:, :, 0, 0, :], lf[:, :, 0, 7, :], lf[:, :, 7, 0, :] ,lf[:, :, 7, 7, :])

    img = None
    for i in range(4):
        if img is None:
            img = plt.imshow(corners[i])
            plt.draw()
            plt.pause(0.3)
        else:
            img.set_data(corners[i])
            plt.draw()
            plt.pause(0.3)

def display_4corner_folder(path):
    files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        lf = read_eslf(f)
        display_4corner(lf)
