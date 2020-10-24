import numpy as np
import torch


device = torch.device('cuda')

def depth_rendering_one(central, disp, v, u, lfsize=[372, 540, 8, 8]):
    b_sz = central.size()[0]
    y_sz = central.size()[1]
    x_sz = central.size()[2]

    y_vals = torch.arange(y_sz, device=device).float()
    x_vals = torch.arange(x_sz, device=device).float()
    y, x = torch.meshgrid([y_vals, x_vals])

    # wrap coordinates by ray depths
    y_t = y + v * disp
    x_t = x + u * disp
    y_t = torch.clamp(y_t, 0, y_sz-1)
    x_t = torch.clamp(x_t, 0, x_sz-1)
    y_t = (y_t/(y_sz-1) - 0.5)*2.0
    x_t = (x_t/(x_sz-1) - 0.5)*2.0

    coord = torch.stack([x_t, y_t], -1)
    coord = coord.repeat([b_sz, 1, 1, 1])

    target = torch.nn.functional.grid_sample(central.permute([0, 3, 1, 2]), coord)
    return target

def mpi_lf_wrapping(rgba, planes, v, u):
    """Render a target view from an MPI representation.
    Args:
      rgba_layers: input MPI [batch, height, width, #planes, 4]
    Returns:
      rendered view [layers, batch, height, width, channels]
    """
    outputs = list()
    for i in np.arange(len(planes)):
        outputs.append(depth_rendering_one(rgba[:,:,:,i,:], planes[i], v, u))
    ret = torch.stack(outputs)
    return ret.permute([0, 1, 3, 4, 2])

def over_composite(rgbas):
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        if i == 0:
            # disable this in softmax mode!
            #output = rgb
            output = rgb * alpha
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output

def mpi_lf_rendering(rgba, planes, v, u):
    target_rgba = mpi_lf_wrapping(rgba, planes, v, u)
    output = over_composite(target_rgba)
    return output
