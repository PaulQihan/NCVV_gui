from xml.dom import xmlbuilder
import torch
import os
import copy
import ipdb
import numpy as np
import scipy
import scipy.fftpack
import math
from ._dct import dct3d

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


frames = [0,3,6,9]
voxel_size = 16
big_data = []


for frame_id in frames:
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']
    print(model_states.keys())

    density, grid_size = split_volume(zero_pads(model_states['density'].cpu(),voxel_size = voxel_size),voxel_size = voxel_size)

    k0, grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(),voxel_size = voxel_size),voxel_size = voxel_size)
    #print(model_states['xyz_min'])

    #density = density.permute(0,2,3,4,1)

    #ipdb.set_trace()
    #mask = torch.nn.functional.softplus(density-4.1).repeat(1,k0.size(1),1,1,1)

    #print(mask.max(), (mask<0.5).sum()/20,(mask>=0.5).sum()/20)
    #k0[mask<0.01] = -9

    big_data.append(torch.cat([density,k0], dim =1))


big_data = torch.cat(big_data)
big_data= big_data.reshape(big_data.size(0),big_data.size(1),-1)
num_voxel_per_volume = big_data.size(0)//len(frames)


print(big_data.size())

num_channels = big_data.size(1)

cnt_mask = big_data[:,0,:]
cnt_mask = torch.nn.functional.softplus(cnt_mask-4.1) >0.4
cnt_mask = cnt_mask.sum(dim=1) #cnt mask 是什么？

thresh = 1
print('valid_voxel',  (cnt_mask>thresh).sum()/cnt_mask.size(0), cnt_mask.size())

print('num_voxel_per_volume',num_voxel_per_volume)
with torch.no_grad():
    #density
    channels = 0
    ratio = 0.2
    tmp_data = big_data[:,channels,:].cuda()
    tmp_data[cnt_mask<=thresh,:] = 0
    U, S, Vh = torch.linalg.svd(tmp_data, full_matrices=False)
    cnt = S.size(0)
    length = int(ratio*cnt)
    recon = U[:, :length] @ torch.diag(S[:length]) @ Vh[:length,:] 
    print('density', ratio, torch.abs(recon - tmp_data).mean().item(), S[length].item(),S[length].item(), U[:, :length].size(),Vh[:length,:].size())
    

    ratio = 0.2
    tmp_data = big_data[:,4:,:].cuda()
    tmp_data[cnt_mask<=1,:,:] = 0
    tmp_data = tmp_data.reshape(tmp_data.size(0),-1)
    U, S, Vh = torch.linalg.svd(tmp_data, full_matrices=False)
    cnt = S.size(0)
   
    length = int(ratio*cnt)
    recon = U[:, :length] @ torch.diag(S[:length]) @ Vh[:length,:] 

    print('feature' , ratio, torch.abs(recon - tmp_data).mean().item(), S[length].item(), U[:, :length].size(),Vh[:length,:].size())
            

print((U[:num_voxel_per_volume,:] - U[num_voxel_per_volume:num_voxel_per_volume*2,:]).abs().mean())