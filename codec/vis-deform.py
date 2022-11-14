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


path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_deform_tv_res_cube'

frame_length=2#00
big_datas = []
masks=[]
for frame_id in range(1,frame_length):
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']

    density = model_states['density'].cpu() # torch.Size([1, 1, 130, 286, 109])
    feature = model_states['k0.k0'].cpu() # torch.Size([1, 12, 130, 286, 109])

    ckpt_deform_path = os.path.join(path, 'fine_last_%d_deform.tar' % frame_id)
    ckpt_deform = torch.load(ckpt_deform_path)
    model_states_deform = ckpt_deform['model_state_dict']
    deformation_field = model_states_deform['deformation_field'].cpu()

    print(density.size())
    print(feature.size())

    cnt_mask = torch.nn.functional.softplus(density - 4.1) > 0.4 # minye的经验值，之后的数据可能需要细调？？？
    masks.append(cnt_mask)

    big_data = torch.cat([density, feature], dim=1)
    big_datas.append(big_data)


big_data = torch.cat(big_datas)
residual=deformation_field

value=torch.linalg.vector_norm(residual,dim=1)
value=value.reshape((value.size(0),-1))

#print(torch.sum(value<0)) 都大于0
print('value', value.size())

masks=torch.cat(masks).bool()
masks=masks.reshape((masks.size(0),-1))




x = torch.linspace(0, big_data.size(2), steps=big_data.size(2))
y = torch.linspace(0, big_data.size(3), steps=big_data.size(3))
z = torch.linspace(0, big_data.size(4), steps=big_data.size(4))
grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (L,L,L,3)

xyz = xyz.reshape((-1, 3))  # (L*L*L,3)

from matplotlib import pyplot as plt
file_path='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_deform_tv_res_cube/hist_deform.png'

v = value[0]
c = v[masks[0]]

print("saving hist image ", file_path)
raw = c.cpu().numpy()
plt.hist(raw.ravel(), log=True)
plt.savefig(file_path)
plt.close()

c = (c-0)/(c.max()-0)
c=get_color(c).squeeze()

xyz_c = xyz[masks[0]]

pts=torch.cat([xyz_c,c], dim=-1).cpu().numpy()

np.savetxt('/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_deform_tv_res_cube/vis_deform_%d.txt' % 0,pts)




