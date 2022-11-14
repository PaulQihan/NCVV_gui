import torch
import os
import math


path = '/usr/data/tmp/mwu/xzq/logs/NHR/xzq_all_fixrgb5'


def log_tensor(data):

    #size = data.size()

    #data = data.reshape(-1)
    mask = data >0
    data[mask] = torch.log(data[mask]+1.0)/math.log(20)
    data[~mask] = -torch.log(-data[~mask]+1.0)/math.log(20)

    #data.reshape(size)

    return data

def ext_volume(volume):
    size_old = list(volume.size()[-3:])
    size = list(volume.size()[-3:])

    for i in range(len(size)):
        size[i] = (16 -size[i]%16) + size[i]

    res = torch.zeros(list(volume.size()[:2]) + size, device = volume.device)
    #print(res.size(), size_old)

    res[:,:,:size_old[0],:size_old[1],:size_old[2]] = volume

    return res

def split_volume(volume):
    volume = ext_volume(volume)
    size = list(volume.size()[-3:])

    for i in range(len(size)):
        size[i] = size[i] // 16

    #print(size)

    voxels = []
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                tmp = volume[:,:,x*16:(x+1)*16, y*16:(y+1)*16, z*16:(z+1)*16]
                voxels.append(tmp)

    voxels = torch.cat(voxels,dim=0)
    #print(voxels.size())

    return voxels




frame_id = 0
for frame_id in range(35):
    ckpt = torch.load(os.path.join(path,'fine_last_%d.tar'% frame_id))



    print('------------bounds--------------')
    print(ckpt['model_state_dict']['xyz_min'])
    print(ckpt['model_state_dict']['xyz_max'])
    print('------------density--------------')
    print(ckpt['model_state_dict']['density'].min())
    print(ckpt['model_state_dict']['density'].max())
    print(torch.abs(ckpt['model_state_dict']['density']).mean())
    
    density = log_tensor(ckpt['model_state_dict']['density'])
    print(density.size())
    density = split_volume(density)

    print('*****')
    print(density.min())
    print(density.max())
    print(torch.abs(density).mean())

    print('------------color--------------')
    print(ckpt['model_state_dict']['k0'].min())
    print(ckpt['model_state_dict']['k0'].max())
    print(torch.abs(ckpt['model_state_dict']['k0']).mean())

    color = log_tensor(ckpt['model_state_dict']['k0'])
    print(color.size())
    color =  split_volume(color)

    print('*****')
    print(color.min())
    print(color.max())
    print(torch.abs(color).mean())


    res = torch.cat([density,color],dim=1)
    print(res.size())
    torch.save(res,'/usr/data/tmp/mwu/data_voxel_xzq/xzq_%d.pth'%frame_id)
