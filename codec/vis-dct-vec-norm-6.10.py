import torch
import os
import copy
import numpy as np

from codec import split_volume,zero_pads,get_color,merge_volume,dct_3d,idct_3d

path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames'

frame_length=2#00
voxel_size = 16
big_datas = []
masks=[]

for frame_id in range(frame_length):
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']

    density, grid_size = split_volume(zero_pads(model_states['density'].cpu(), voxel_size=voxel_size),
                                      voxel_size=voxel_size)
    k0, grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(), voxel_size=voxel_size), voxel_size=voxel_size)

    big_data = torch.cat([density, k0], dim=1)
    big_data = big_data.reshape(big_data.size(0), big_data.size(1), -1)

    thresh = 1
    cnt_mask = big_data[:, 0, :]
    cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
    cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？

    masks.append(cnt_mask)

    big_datas.append(big_data)

big_data = torch.stack(big_datas)
residual=(big_data[1:]-big_data[:-1]) # torch.Size([frame_num, 1134, 13, 4096])

residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)


masks=torch.stack(masks)
masks=(masks[1:] + masks[:-1]).bool()
masks=masks.reshape((masks.size(0),-1))


x = torch.linspace(0, voxel_size, steps=voxel_size)
y = torch.linspace(0, voxel_size, steps=voxel_size)
z = torch.linspace(0, voxel_size, steps=voxel_size)
grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (L,L,L,3)

xyz = xyz.reshape((-1, 3))  # (L*L*L,3)



os.makedirs('/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/vis_dct',exist_ok=True)
for frameid in range(frame_length - 1):
    residual_cur = residual[frameid]
    RES = dct_3d(residual_cur)  # torch.Size([1134, 13, 16, 16, 16])
    RES_norm = torch.linalg.vector_norm(RES, dim=1)
    #print(masks[frameid].size())
    c = RES_norm[masks[frameid]]
    c=torch.mean(c,dim=0).reshape((-1))
    # print(c.size())
    # print(c[0])
    # print(c.min(),c.max())
    # exit()
    c = (c-c.min())/(c.max()-c.min())
    c=get_color(c).squeeze()

    #xyz_c = xyz[masks[frameid]]

    pts=torch.cat([xyz,c], dim=-1).cpu().numpy()

    np.savetxt('/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/vis_dct/vis_%d.txt' % frameid,pts)




