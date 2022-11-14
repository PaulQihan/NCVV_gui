import torch
import os
import copy
import numpy as np

from codec import split_volume,zero_pads,zero_unpads,get_origin_size,get_color,merge_volume,dct_3d,idct_3d
from codec import gradient_compression,gen_3d_quant_tbl,quant_norm,quant_norm_int12,quant_norm_int16,quant,anal_res
from codec import encode_int8,encode_hqh, decode_hqh,quantize_quality,encode_jpeg,decode_jpeg

from matplotlib import pyplot as plt

def jpeg_stoch(data, dataname):
    data = data.cpu().numpy()
    data=np.rint(data)
    tmp = torch.Tensor(data)
    print(f"zero percent {dataname}: ", torch.sum(tmp == 0) / tmp.nelement())

path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024'
# vis_path_density = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_nonorm_hist_density.png'
# vis_path_color = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_nonorm_hist_color.png'
#path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames_sigmoid'
# vis_path_density = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames_sigmoid/quant_nonorm_hist_density.png'
# vis_path_color = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames_sigmoid/quant_nonorm_hist_color.png'

expr_name = '/quant_quality'
qualities = (99,100,98,96,94,92,90,80,50,20,10,2)

QTY_3d =gen_3d_quant_tbl()

frame_length=2#00
voxel_size = 8
big_datas = []
masks=[]

x = torch.linspace(0, voxel_size, steps=voxel_size)
y = torch.linspace(0, voxel_size, steps=voxel_size)
z = torch.linspace(0, voxel_size, steps=voxel_size)
grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (L,L,L,3)

xyz = xyz.reshape((-1, 3))  # (L*L*L,3)

thresh = 1

quant_type="jpeg"

for frame_id in range(frame_length):
    print('process frame', frame_id)
    ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
    ckpt = torch.load(ckpt_path)
    model_states = ckpt['model_state_dict']
    try:
        rgbfeat_sigmoid=ckpt['model_kwargs']['rgbfeat_sigmoid']
    except:
        rgbfeat_sigmoid=False

    # pure_model = torch.cat([model_states['density'], model_states['k0.k0']], dim=1).clone().cpu()
    # torch.save(pure_model, '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/pure_model.tar')
    # exit()

    density, grid_size = split_volume(zero_pads(model_states['density'].cpu(), voxel_size=voxel_size),
                                      voxel_size=voxel_size)
    k0, grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(), voxel_size=voxel_size), voxel_size=voxel_size)

    print("density min", density.min())
    print("density max", density.max())
    density=density.cuda()
    k0=k0.cuda()

    if rgbfeat_sigmoid:
        print("sigmoid after rgbfeat")
        k0=torch.sigmoid(k0)

    big_data = torch.cat([density, k0], dim=1)
    big_data = big_data.reshape(big_data.size(0), big_data.size(1), -1)


    cnt_mask = big_data[:, 0, :]
    cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
    cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？

    masks.append(cnt_mask)

    big_datas.append(big_data)

    if frame_id==0:

        continue

    big_data = torch.stack(big_datas)
    residual=(big_data[1:]-big_data[:-1]) # torch.Size([frame_num, 1134, 13, 4096])

    residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)


    masks=torch.stack(masks)
    masks=(masks[1:] + masks[:-1]).bool()
    masks=masks.reshape((masks.size(0),-1))

    residual_cur = residual[0]
    residual_rec_o = torch.zeros_like(residual_cur,device=residual_cur.device)
    # if rgbfeat_sigmoid:
    #     residual_rec[: ,1:]+=0.5
    residual_cur=residual_cur[masks[0]]

    # density_RES,_,_=encode_int8(residual_cur[:, :1],QTY_3d)
    # color_RES, _, _ = encode_int8(residual_cur[:, 1:], QTY_3d)
    for j, q in enumerate(qualities):

        # RES = dct_3d(residual_cur, norm='ortho')
        #
        # quant_table=quantize_quality(QTY_3d,q)
        # RES_quant=RES/quant_table
        # RES_quant=torch.round(RES_quant)
        # RES = RES_quant * quant_table
        # residual_rec_dct = idct_3d(RES, norm='ortho')
        bits, header = encode_jpeg(residual_cur, q)

        with open(os.path.join(path, f'compressed_{q}.ncrf'), 'wb') as compressed_file:
            bits.tofile(compressed_file)

        with open(os.path.join(path, f'compressed_{q}.ncrf'), 'rb') as compressed_file:
            residual_rec_dct = decode_jpeg(compressed_file, header, residual_cur.device)

        print("quality ", q)
        residual_rec = torch.zeros_like(residual_rec_o, device=residual_cur.device)
        residual_rec[masks[0]]=residual_rec_dct

        residual_rec=residual_rec.reshape(residual.size(0),residual.size(1),residual.size(2),-1)
        rec_feature=big_datas[0]+residual_rec
        rec_feature = rec_feature.reshape(residual.size(1), residual.size(2), voxel_size,voxel_size,voxel_size)

        rec_feature=merge_volume(rec_feature,grid_size)
        origin_size = get_origin_size(model_states['k0.k0'])
        rec_feature=zero_unpads(rec_feature,origin_size)

        density_rec=rec_feature[:1]
        feature_rec=rec_feature[1:]

        model_states['density']=density_rec.unsqueeze(0).cpu().clone()
        model_states['k0.k0'] = feature_rec.unsqueeze(0).cpu().clone()

        ckpt['model_state_dict']=model_states
        ckpt_path = os.path.join(path, f'fine_last_{frame_id}_recq_{q}_quality.tar')

        torch.save(ckpt,ckpt_path)
    #
    masks=[masks[0]]
    big_datas=[big_datas[0]]


