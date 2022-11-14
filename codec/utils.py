import torch
import os
import copy
import numpy as np
import math
import matplotlib.pyplot as plt

def torch_and(a,b):
    return (a*b).bool()

def get_color(sdf): #gt do not need
    '''从蓝色渐变到红色
    RGB分量，分别为0，0，255，B分量不变，G增加到255，到达结果RGB分别为0，255，255；
    G分量不变，B分量减小到0，到达结果RGB分别为0，255，0；
    GB分量都不变，R增加到255，到达结果RGB分别为255，255，0；
    RB分量都不变，G减小到0，到达结果255，0，0
    '''
    color=torch.zeros((1,sdf.size(0),3))#.cuda()

    color[...,torch_and(sdf>0, sdf<0.25),2 ]=255
    color[..., torch_and(sdf>0, sdf<0.25), 1] = (sdf[torch_and(sdf>0, sdf<0.25)]-0.1)/0.25*255

    color[...,torch_and(sdf>=0.25 , sdf<0.75),1]=255
    color[..., torch_and(sdf>=0.25 , sdf<0.75), 2]= (0.5-sdf[torch_and(sdf>=0.25 , sdf<0.75)])/0.25*255

    color[...,torch_and(sdf >= 0.5 , sdf < 0.75), 0] = (sdf[torch_and(sdf >= 0.5 , sdf < 0.75)] - 0.5) / 0.25 * 255

    color[..., sdf >= 0.75, 0] = 255
    color[..., torch_and(sdf >= 0.75 ,sdf < 1), 1] = (1 - sdf[torch_and(sdf >= 0.75 ,sdf < 1)]) / 0.25 * 255

    return color

def get_origin_size(data):
    size=list(data.size()) #[1, 12, 130, 286, 109]
    return size[2:]

def zero_pads(data, voxel_size=16):
    if data.size(0) == 1:
        data = data.squeeze(0)

    size = list(data.size())

    new_size = copy.deepcopy(size)
    for i in range(1, len(size)):
        if new_size[i] % voxel_size == 0:
            continue
        new_size[i] = (new_size[i] // voxel_size + 1) * voxel_size

    res = torch.zeros(new_size, device=data.device)
    res[:, :size[1], :size[2], :size[3]] = data.clone()
    return res


def zero_unpads(data, size):
    return data[:, :size[0], :size[1], :size[2]]


def split_volume(data, voxel_size=16):
    size = list(data.size()) #[12, 144, 288, 112]
    for i in range(1, len(size)):
        size[i] = size[i] // voxel_size

    res = []
    for x in range(size[1]):
        for y in range(size[2]):
            for z in range(size[3]):
                res.append(data[:, x * voxel_size:(x + 1) * voxel_size, y * voxel_size:(y + 1) * voxel_size,
                           z * voxel_size:(z + 1) * voxel_size].clone())

    res = torch.stack(res)

    return res, size[1:]


def merge_volume(data, size):
    M, NF, Vx, Vy, Vz = data.shape  
    data_tmp = data[:size[0]*size[1]*size[2]].reshape(size[0],size[1],size[2],NF,Vx,Vy,Vz) 
    data_tmp = data_tmp.permute(3,0,4,1,5,2,6) 
    res = data_tmp.reshape(NF, size[0]*Vx, size[1]*Vy, size[2]*Vz) 
    return res  



def gradient_compression(RES,threshold=500):
    #RES torch.Size([1134, 13, 16, 16, 16])
    # RES_norm=torch.norm(RES,dim=1)
    # RES_norm=RES_norm.reshape((RES.size(0),-1))
    # print(torch.max(RES_norm,dim=1))
    # 这个threshold 是否应该每个voxel不同
    res_size=RES.size()
    voxel_size=res_size[-1]
    dRes_x=RES[:,:,1:]-RES[:,:,:-1]
    dRes_y=RES[:,:,:,1:]-RES[:,:,:,:-1]
    dRes_z=RES[:,:,:,:,1:]-RES[:,:,:,:,:-1]
    #pad
    dRes_x=torch.cat([dRes_x,dRes_x[:,:,-1:]],dim=2)
    dRes_y=torch.cat([dRes_y,dRes_y[:,:,:,-1:]],dim=3)
    dRes_z=torch.cat([dRes_z,dRes_z[:,:,:,:,-1:]],dim=4)

    grad=torch.stack([dRes_x,dRes_y,dRes_z],dim=2)
    grad_norm=torch.norm(grad, dim=2)

    RES_voxel=RES.reshape(RES.size(0),RES.size(1),-1)
    grad_norm_voxel=grad_norm.reshape(RES.size(0),RES.size(1),-1)
    mask=grad_norm_voxel<500
    RES_voxel[~mask]=0
    #mask_num=torch.sum(mask, dim=-1)
    RES_mean=torch.sum(RES_voxel,dim=-1)/torch.sum(mask,dim=-1)
    print(res_size)
    print(RES_mean.size())
    RES_mean=RES_mean.reshape(RES_mean.size(0),RES_mean.size(1),1,1,1)
    RES_mean=RES_mean.expand(-1,-1,voxel_size,voxel_size,voxel_size)
    mask=mask.reshape(RES_mean.size(0),RES_mean.size(1),voxel_size,voxel_size,voxel_size)
    RES[mask]=RES_mean[mask]

    print(grad_norm[mask].size())
    #grad_median=torch.median(grad_norm[mask])
    #vgrad_normis
    raw =grad_norm.cpu().numpy()
    plt.hist(np.minimum(500,raw.ravel()), log=True)
    plt.savefig('/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/grad_hist.png')
    print("min", torch.min(grad_norm))
    print("non zero percentage ", torch.sum(grad_norm>500)/grad_norm.numel() )

    #cal compression rate


    return RES


