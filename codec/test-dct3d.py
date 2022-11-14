from xml.dom import xmlbuilder
import torch
import os
import copy
import ipdb
import numpy as np
import scipy
import scipy.fftpack
import math
from codec import dct_3d,idct_3d


def zero_pads(data, voxel_size =16):
    if data.size(0)==1:
        data = data.squeeze(0)

    size = list(data.size())

    new_size = copy.deepcopy(size)
    for i in range(1,len(size)):
        if new_size[i]%voxel_size==0:
            continue
        new_size[i] = (new_size[i]//voxel_size+1)*voxel_size
    
    res= torch.zeros(new_size, device = data.device)
    res[:,:size[1],:size[2],:size[3]] = data.clone()
    return res

def zero_unpads(data, size):
    
    return data[:,:size[0],:size[1],:size[2]]


def split_volume(data, voxel_size =16):
    size = list(data.size())
    for i in range(1,len(size)):
        size[i] = size[i]//voxel_size

    res = []
    for x in range(size[1]):
        for y in range(size[2]):
            for z in range(size[3]):
                res.append(data[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size].clone())

    res = torch.stack(res)

    return res,size[1:]


def merge_volume(data,size):
    M, NF, Vx, Vy, Vz = data.shape  
    data_tmp = data[:size[0]*size[1]*size[2]].reshape(size[0],size[1],size[2],NF,Vx,Vy,Vz) 
    data_tmp = data_tmp.permute(3,0,4,1,5,2,6) 
    res = data_tmp.reshape(NF, size[0]*Vx, size[1]*Vy, size[2]*Vz) 
    return res  



path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames'


frame_length=200
voxel_size = 16
big_datas = []


for frame_id in range(frame_length):
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']


    density, grid_size = split_volume(zero_pads(model_states['density'].cpu(),voxel_size = voxel_size),voxel_size = voxel_size)

    k0, grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(),voxel_size = voxel_size),voxel_size = voxel_size)

    #set density to zero

    big_data=torch.cat([density,k0], dim =1)

    big_data = big_data.reshape(big_data.size(0), big_data.size(1),  -1)

    # mask density 接近于0的voxel
    thresh = 1
    cnt_mask = big_data[:, 0, :]
    cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
    cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？

    big_data[cnt_mask <= thresh, :, :] = 0

    print('valid_voxel', (cnt_mask > thresh).sum() / cnt_mask.size(0), cnt_mask.size())

    big_datas.append(big_data)


big_data = torch.stack(big_datas)

residual=(big_data[1:]-big_data[:-1])

print("residual mean", torch.mean(residual))

residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)

#应该scale到0-1
with torch.no_grad():
    for i in range(frame_length-1):
        residual_cur=residual[i].cuda()
        RES=dct_3d(residual_cur) #torch.Size([1134, 13, 16, 16, 16])
        residual_rec=idct_3d(RES)

        print("error sum", (torch.abs(residual_cur - residual_rec)).sum())
        print("error mean", (torch.abs(residual_cur - residual_rec)).mean())



