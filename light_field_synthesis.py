import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
import torch.optim as optim
from PIL import Image
import skimage
import skimage.transform
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import mpi
from vgg import vgg_loss, weighted_vgg_loss


device = torch.device('cuda')
feature_extract = True
batch_size = 1
train_batch_size = 8
num_mpi_planes = 8
lfsize = [372, 540, 8, 8]
lfsize_train = [256, 256, 8, 8]
lfsize_test = [368, 528, 8, 8]
T = np.load('color_transfer.npy')
Run_stats = True 
AFF = True  # stop updating batch norm layers
MOM = 0.0
mode = 'validate'
#resume = True


def normalize_lf(lf):
    return 2.0*(lf-0.5)

def denormalize_lf(lf):
    return lf/2.0+0.5

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class MpiNet(nn.Module):
    def __init__(self, ngf=32, num_outputs=num_mpi_planes*5):
        super(MpiNet, self).__init__()

        # rgba network
        self.conv1_1 = nn.Conv2d(4, ngf, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(ngf, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv1_2 = nn.Conv2d(ngf, ngf*2, 3, padding=1, stride=2)
        self.bn1_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv2_1 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv2_2 = nn.Conv2d(ngf*2, ngf*4, 3, padding=1, stride=2)
        self.bn2_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv3_1 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv3_2 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv3_3 = nn.Conv2d(ngf*4, ngf*8, 3, padding=1, stride=2)
        self.bn3_3 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv4_1 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_1 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv4_2 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_2 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv4_3 = nn.Conv2d(ngf*8, ngf*8, 3, padding=2, dilation=2)
        self.bn4_3 = nn.BatchNorm2d(ngf*8, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv6_1 = nn.ConvTranspose2d(ngf*16, ngf*4, 4, padding=1, stride=2)
        self.bn6_1 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv6_2 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv6_3 = nn.Conv2d(ngf*4, ngf*4, 3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(ngf*4, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv7_1 = nn.ConvTranspose2d(ngf*8, ngf*2, 4, padding=1, stride=2)
        self.bn7_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv7_2 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        
        self.conv8_1 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, padding=1, stride=2)
        self.bn8_1 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv8_2 = nn.Conv2d(ngf*2, ngf*2, 3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(ngf*2, track_running_stats=Run_stats, affine=AFF, momentum=MOM)
        self.conv8_3 = nn.Conv2d(ngf*2, num_outputs, 1)

    def forward(self, x, depth, v, u):
        # deeplens depth
        depth_features = depth.permute([0, 3, 1, 2])

        # get image planes
        c1_1 = self.conv1_1(torch.cat((x.permute([0, 3, 1, 2]), depth_features), 1))
        c1_1 = self.bn1_1(c1_1)
        c1_1 = F.relu(c1_1)
        c1_2 = self.conv1_2(c1_1)
        c1_2 = self.bn1_2(c1_2)
        c1_2 = F.relu(c1_2)
        c2_1 = self.conv2_1(c1_2)
        c2_1 = self.bn2_1(c2_1)
        c2_1 = F.relu(c2_1)
        c2_2 = self.conv2_2(c2_1)
        c2_2 = self.bn2_2(c2_2)
        c2_2 = F.relu(c2_2)
        c3_1 = self.conv3_1(c2_2)
        c3_1 = self.bn3_1(c3_1)
        c3_1 = F.relu(c3_1)
        c3_2 = self.conv3_2(c3_1)
        c3_2 = self.bn3_2(c3_2)
        c3_2 = F.relu(c3_2)
        c3_3 = self.conv3_3(c3_2)
        c3_3 = self.bn3_3(c3_3)
        c3_3 = F.relu(c3_3)
        c4_1 = self.conv4_1(c3_3)
        c4_1 = self.bn4_1(c4_1)
        c4_1 = F.relu(c4_1)
        c4_2 = self.conv4_2(c4_1)
        c4_2 = self.bn4_2(c4_2)
        c4_2 = F.relu(c4_2)
        c4_3 = self.conv4_3(c4_2)
        c4_3 = self.bn4_3(c4_3)
        c4_3 = F.relu(c4_3)
        
        c6_1 = self.conv6_1(torch.cat((c4_3, c3_3), 1))
        c6_1 = self.bn6_1(c6_1)
        c6_1 = F.relu(c6_1)
        c6_2 = self.conv6_2(c6_1)
        c6_2 = self.bn6_2(c6_2)
        c6_2 = F.relu(c6_2)
        c6_3 = self.conv6_3(c6_2)
        c6_3 = self.bn6_3(c6_3)
        c6_3 = F.relu(c6_3)
        c7_1 = self.conv7_1(torch.cat((c6_3, c2_2), 1))
        c7_1 = self.bn7_1(c7_1)
        c7_1 = F.relu(c7_1)
        c7_2 = self.conv7_2(c7_1)
        c7_2 = self.bn7_2(c7_2)
        c7_2 = F.relu(c7_2)
        c8_1 = self.conv8_1(torch.cat((c7_2, c1_2), 1))
        c8_1 = self.bn8_1(c8_1)
        c8_1 = F.relu(c8_1)
        c8_2 = self.conv8_2(c8_1)
        c8_2 = self.bn8_2(c8_2)
        c8_2 = F.relu(c8_2)
        c8_3 = self.conv8_3(c8_2)
        c8_3 = torch.tanh(c8_3)

        rgba_layers = torch.reshape(c8_3.permute([0, 2, 3, 1]), (batch_size, lfsize[0], lfsize[1], num_mpi_planes, 5))
        color_layers = rgba_layers[:, :, :, :, :3]
        alpha_layers = rgba_layers[:, :, :, :, 3:4]
        depth_layers = rgba_layers[:, :, :, :, 4]
        # Rescale alphas to (0, 1)
        alpha_layers = (alpha_layers + 1.) / 2.
        rgba_layers = torch.cat((color_layers, alpha_layers), 4)
        rgbad_layers = torch.cat((color_layers, alpha_layers, torch.unsqueeze(depth_layers, dim=-1)), 4)

        depth_planes = depth_layers.mean(1).mean(1)

        # rendering
        output = list()
        for i in range(rgba_layers.shape[0]):
            output.append(mpi.mpi_lf_rendering(rgba_layers[i:i+1,...], depth_planes[i], v, u))
        output = torch.cat(output, dim=0)

        return output, rgbad_layers

    def load_network(self, filename):
        model = torch.load(filename)
        return model
    
    def save_network(self, network, model_path):
        torch.save(network.cpu().state_dict(), model_path)
        model.to(device=device)
        return True

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

def lock_mpi_net(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'inp' not in name:
            param.requires_grad = False


def color_transform(src, T):
    """
    src: [H, W, 3]
    T: [3, 256]
    """
    res = np.zeros_like(src, dtype=np.float)
    for i in range(3):
        for j in range(src.shape[0]):
            for k in range(src.shape[1]):
                res[j][k][i] = T[i,src[j,k,i]]*1.0
                if res[j,k,i] < 0:
                    res[j,k,i] = 0
                if res[j,k,i] > 1:
                    res[j,k,i] = 1
    res = res*255.0
    return res.astype(np.uint8)

def data_augmentation(x, y, mode='train'):
    """
    x, y: np array, [B, C, H, W]
    """
    x_list = list()
    y_list = list()
    for i in range(x.shape[0]):
        # gamma correction
        xi = skimage.exposure.adjust_gamma(x[i,...], 0.4)
        yi = skimage.exposure.adjust_gamma(y[i,...], 0.4)

        # color transfer
        if mode == 'train':
            xi = color_transform(xi, T)
            yi = color_transform(yi, T)
        
        xi = Image.fromarray(xi)
        yi = Image.fromarray(yi)

        if mode == 'train':
            pick = random.randint(0, 4)
            if pick == 0:
                # random brightness
                brightness_factor = 1.0 + random.uniform(0, 0.3)
                xi = ttf.adjust_brightness(xi, brightness_factor)
                yi = ttf.adjust_brightness(yi, brightness_factor)
            elif pick == 1:
                # random saturation
                saturation_factor = 1.0 + random.uniform(-0.2, 0.5)
                xi = ttf.adjust_saturation(xi, saturation_factor)
                yi = ttf.adjust_saturation(yi, saturation_factor)
            elif pick == 2:
                # random hue
                hue_factor = random.uniform(-0.2, 0.2)
                xi = ttf.adjust_hue(xi, hue_factor)
                yi = ttf.adjust_hue(yi, hue_factor)
            elif pick == 3:
                # random contrast
                contrast_factor = 1.0 + random.uniform(-0.2, 0.4)
                xi = ttf.adjust_contrast(xi, contrast_factor)
                yi = ttf.adjust_contrast(yi, contrast_factor)
            elif pick == 4:
                # random swap color channel
                permute = np.random.permutation(3)
                xi = np.array(xi)
                yi = np.array(yi)
                xi = xi[..., permute]
                yi = yi[..., permute]
        xi = np.clip(np.array(xi)/255.0, 0, 1.0)
        yi = np.clip(np.array(yi)/255.0, 0, 1.0)
        x_list.append(xi)
        y_list.append(yi)
    x_ret = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float)
    y_ret = torch.tensor(np.stack(y_list, axis=0), dtype=torch.float)
    x_ret = normalize_lf(x_ret)
    y_ret = normalize_lf(y_ret)

    return x_ret.to(device), y_ret.to(device)

# dataset
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
        
        # crop
        if self.mode == 'train':
            a = random.randint(0, 372-lfsize_train[0])
            b = random.randint(0, 540-lfsize_train[1])
            lf = lf[a:a+lfsize_train[0], :, b:b+lfsize_train[1], :, :]
            base_path = ''  
        else:
            a, b = 0, 0
            lf = lf[a:a+lfsize_test[0], :, b:b+lfsize_test[1], :, :]
            base_path = 'data/depth/' 
        lf = np.transpose(lf, (0, 2, 1, 3, 4))
        aif = lf[:, :, lfsize[2]//2, lfsize[3]//2, :]

        # get deeplens depth
        filename = self.filenames[idx].split('\\')[-1]
        file_idx = filename.split('.')[0]
        depth = np.load(base_path + file_idx + '_depth.npy')
        depth = np.expand_dims(depth, axis=-1)
        if self.mode == 'train':
            depth = depth[a:a+lfsize_train[0], b:b+lfsize_train[1], :]
        else:
            depth = depth[a:a+lfsize_test[0], b:b+lfsize_test[1], :]
        depth = torch.tensor(depth, dtype=torch.float)
        
        return aif, lf, depth


def gradient_hook_deco(mask):
    def hook(grad):
        return mask*grad
    return hook

def get_mask(rgbad, v, u):
    color_layers = rgbad[:, :, :, :, :3]
    alpha_layers = rgbad[:, :, :, :, 3:4]
    depth_layers = rgbad[:, :, :, :, 4]
    depth_planes = depth_layers.mean(1).mean(1)

    alpha = alpha_layers[:,:,:,:,0]
    a_ini = alpha.permute([0, 3, 1, 2])

    rgba_sr = torch.tensor(rgbad[...,:4])
    for i in np.arange(0, num_mpi_planes):
        # calculate a_occ_i:
        for j in range(i+1, num_mpi_planes):
            if j == i+1:
                a_occ_i = a_ini[:,j:j+1,:,:].clone().detach().requires_grad_(True)
            else:
                a_occ_i = a_occ_i*(1-a_ini[:,j:j+1,:,:]) + a_ini[:,j:j+1,:,:]
        if i+1 == num_mpi_planes:
            a_occ_i = torch.zeros_like(a_ini[:,0:1,:,:], requires_grad=True)

        a_occ_i = a_occ_i.permute([0, 2, 3, 1])
        rgba_sr[:,:,:,i,:] = a_ini[:,i:i+1,:,:].permute([0,2,3,1])*(1-a_occ_i)

    target_rgba = mpi.mpi_lf_wrapping(rgba_sr.cuda(), depth_planes[0], v, u)
    target_alpha = target_rgba[:,:,:,:,3:]
    target_alpha_sum = torch.sum(target_alpha, dim=0)
    target_alpha_sum = torch.clamp(target_alpha_sum, 0, 1)
    weight = 1. - target_alpha_sum
    weight[weight<0.2] = 0

    return weight

def lf_loss(lf_shear, labels, v, u, rgbad):
    """
    Args:
        lf_shear: [B, H, W, C]
        labels: [B, H, W, C]
    """
    shear_loss = torch.mean(torch.abs(denormalize_lf(lf_shear) - denormalize_lf(labels)))
    #beta = 0.005
    #vgg_loss = beta*lf_weighted_loss(lf_shear, labels, v, u, rgbad_init)

    return shear_loss + depth_loss #+ vgg_loss

def lf_loss_l1(lf_shear, labels):
    """
    Args:
        lf_shear: [B, H, W, C]
        labels: [B, H, W, C]
    """
    shear_loss = torch.mean(torch.abs(denormalize_lf(lf_shear) - denormalize_lf(labels)))
    return shear_loss


def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)

def change_param_to_eval():
    global lfsize, batch_size
    lfsize[0] = lfsize_test[0]
    lfsize[1] = lfsize_test[1]
    batch_size = 1

def change_param_to_train():
    global lfsize, batch_size
    lfsize[0] = lfsize_train[0]
    lfsize[1] = lfsize_train[1]
    batch_size = train_batch_size


def evaluate_model(model_vgg, model_init, val_loader):
    model_init.eval()
    model_vgg.eval()
    change_param_to_eval()
    total_loss = 0.0

    for i_batch, batched_inputs in enumerate(val_loader):
        labels = batched_inputs[1]
        depth = batched_inputs[2]

        with torch.set_grad_enabled(False):
            lf = np.zeros((1, 368, 528, 8, 8, 3))
            gt = np.zeros((1, 368, 528, 8, 8, 3))
            loss = 0.0

            for v in range(0, 8):
                for u in range(0, 8):
                    inputs = labels[:,:,:,4,4,:]
                    target = labels[:,:,:,v,u,:]
                    inputs, target = data_augmentation(inputs.numpy(), target.numpy(), mode='val')

                    lf_shear_init, rgbad_init = model_fix(inputs, depth.to(device), v-4, u-4)
                    mask = get_mask(rgbad_init.detach(), v-4, u-4)
                    mask_rest = 1.0 - mask
                    lf_shear2, rgbad2 = model_vgg(inputs, depth.to(device), v-4, u-4)
                    final_output = mask_rest*lf_shear_init + mask*lf_shear2

                    loss1 = lf_loss_l1(final_output, target)
                    loss += loss1
                    lf[:,:,:,v,u,:] = final_output.cpu().numpy()
                    gt[:,:,:,v,u,:] = target.cpu().numpy()
            
            np.save('results/output_%d.npy'%(i_batch), denormalize_lf(lf))
            np.save('results/rgbad_%d.npy'%(i_batch), rgbad_init.cpu().numpy())


def train_model(model_vgg, model_fix, train_loader, val_loader, optimizer2, num_epochs=25):
    model_fix.eval()

    iters = 0
    accu_train_loss = 0.0
    for epoch in range(num_epochs):

        # Iterate over data.
        change_param_to_train()
        batch_count = 0
        for i_batch, batched_inputs in enumerate(train_loader):
            pos = [(0, 0), (0, 7), (7, 0), (7, 7)]
            c = random.randint(0, 3)
            v = random.randint(0, 7)
            u = random.randint(0, 7)

            labels = batched_inputs[1]
            inputs = labels[:,:,:,pos[c][0],pos[c][1],:]
            target = labels[:,:,:,v,u,:]
            # data augmentation
            inputs, target = data_augmentation(inputs.numpy(), target.numpy())
            depth = batched_inputs[2]

            # zero the parameter gradients
            optimizer2.zero_grad()
            with torch.set_grad_enabled(True):

                # calculate the gradient mask
                lf_shear_init, rgbad_init = model_fix(inputs, depth[:,c,:,:,:].to(device), v-pos[c][0], u-pos[c][1])
                mask = get_mask(rgbad_init.detach(), v, u)
                mask_rest = 1.0 - mask

                #lf_shear, rgbad = model(inputs, depth[:,c,:,:,:].to(device), v-pos[c][0], u-pos[c][1])
                lf_shear2, rgbad2 = model_vgg(inputs, depth[:,c,:,:,:].to(device), v-pos[c][0], u-pos[c][1])
                final_output = mask_rest*lf_shear_init + mask*lf_shear2

                hook = lf_shear2.register_hook(gradient_hook_deco(mask))
                v_loss = 1.0*vgg_loss(normalize_batch(denormalize_lf(final_output).permute([0, 3, 1, 2])), normalize_batch(denormalize_lf(target).permute([0, 3, 1, 2])))
                v_loss.backward()
                hook.remove()
                optimizer2.step()

            # statistics
            cur_loss = v_loss.item()
            accu_train_loss += cur_loss
            print('epoch %d, iter %d, loss: %lf' %(epoch, i_batch+1, cur_loss), end="\r")

            if (iters+1)%400 == 0:
                summary_writer.add_scalar('train_loss', accu_train_loss/400, (iters+1))
                accu_train_loss = 0.0

                save_checkpoint({'state_dict': model_vgg.state_dict(), 'optimizer' : optimizer2.state_dict()}, filename='checkpoints/occluded_%d.pth.tar'%(iters+1))

                # recover to train mode
                model_vgg.train()
                change_param_to_train()
            iters += 1
    return 'finished'

# create the model
model_vgg = MpiNet()
model_fix = MpiNet()

model_vgg = model_vgg.to(device=device)
model_fix = model_fix.to(device=device)

# create the optimizer
'''
# stage 1, train the visible network
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
optimizer = optim.Adam(params_to_update, lr=0.0002) # Observe that all parameters are being optimized
'''
params_to_update2 = []
for name, param in model_vgg.named_parameters():
    if param.requires_grad == True:
        params_to_update2.append(param)
optimizer2 = optim.Adam(params_to_update2, lr=0.0002)

if mode == 'train':
    summary_writer = SummaryWriter('logs/')


if __name__ == '__main__':
    validation_set = LightFieldDataset('data/lf', mode='validate')
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    checkpoint = torch.load('checkpoints/occluded_net.pth.tar')
    model_vgg.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('checkpoints/visible_net.pth.tar')
    model_fix.load_state_dict(checkpoint['state_dict'])
    evaluate_model(model_vgg, model_fix, val_loader)
