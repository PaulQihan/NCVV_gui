import torch
import os
import copy
import numpy as np

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
    color[..., torch_and(sdf>0, sdf<0.25), 1] = (sdf[torch_and(sdf>0, sdf<0.25)])/0.25*255

    color[...,torch_and(sdf>=0.25 , sdf<0.75),1]=255
    color[..., torch_and(sdf>=0.25 , sdf<0.75), 2]= (0.5-sdf[torch_and(sdf>=0.25 , sdf<0.75)])/0.25*255

    color[...,torch_and(sdf >= 0.5 , sdf < 0.75), 0] = (sdf[torch_and(sdf >= 0.5 , sdf < 0.75)] - 0.5) / 0.25 * 255

    color[..., sdf >= 0.75, 0] = 255
    color[..., torch_and(sdf >= 0.75 ,sdf < 1), 1] = (1 - sdf[torch_and(sdf >= 0.75 ,sdf < 1)]) / 0.25 * 255

    return color


path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames'

frame_length=2#00
big_datas = []
masks=[]
for frame_id in range(frame_length):
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']

    density = model_states['density'].cpu() # torch.Size([1, 1, 130, 286, 109])
    feature = model_states['k0.k0'].cpu() # torch.Size([1, 12, 130, 286, 109])

    print(density.size())
    print(feature.size())

    cnt_mask = torch.nn.functional.softplus(density - 4.1) > 0.4 # minye的经验值，之后的数据可能需要细调？？？
    masks.append(cnt_mask)

    big_data = torch.cat([density, feature], dim=1)
    big_datas.append(big_data)


big_data = torch.cat(big_datas)
residual=(big_data[1:]-big_data[:-1])

value=torch.linalg.vector_norm(residual,dim=1)
value=value.reshape((value.size(0),-1))

print('value', value.size())

masks=torch.cat(masks)
masks=(masks[1:] + masks[:-1]).bool()
masks=masks.reshape((masks.size(0),-1))


x = torch.linspace(0, big_data.size(2), steps=big_data.size(2))
y = torch.linspace(0, big_data.size(3), steps=big_data.size(3))
z = torch.linspace(0, big_data.size(4), steps=big_data.size(4))
grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (L,L,L,3)

xyz = xyz.reshape((-1, 3))  # (L*L*L,3)

for frameid in range(frame_length - 1):

    v = value[frameid]
    c = v[masks[frameid]]
    c = (c-c.min())/(c.max()-c.min())
    c=get_color(c).squeeze()

    xyz_c = xyz[masks[frameid]]

    pts=torch.cat([xyz_c,c], dim=-1).cpu().numpy()

    np.savetxt('/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/vis_new_%d.txt' % frameid,pts)




