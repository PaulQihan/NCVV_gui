import sys
sys.path.append('/data/new_disk2/wangla/tmp/NCVV')
import torch
import os
import copy
import numpy as np

from codec import split_volume,zero_pads,zero_unpads,get_origin_size,get_color,merge_volume
from codec import gradient_compression,gen_3d_quant_tbl,quant_norm,quant_norm_int12,quant_norm_int16,quant,anal_res
from codec import encode_int8,encode_hqh, decode_hqh,quantize_quality,encode_jpeg,decode_jpeg,Timer
import time
from bitarray import bitarray
import json

# init enviroment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-2'

qualitys=[99]
for quality in qualitys:
    expr_name = f'dynamic+{quality}'
    os.makedirs(os.path.join(path,expr_name),exist_ok=True)

    frame_length=200
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
        while not os.path.exists(ckpt_path):
            torch.cuda.empty_cache()
            time.sleep(3000)
            print("waiting checkpoint ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model_states = ckpt['model_state_dict']

        density, grid_size = split_volume(zero_pads(model_states['density'].cpu(), voxel_size=voxel_size),
                                          voxel_size=voxel_size)
        if frame_id == 0:
            k0, grid_size = split_volume(zero_pads((model_states['k0.k0']).cpu(), voxel_size=voxel_size), voxel_size=voxel_size)

        else:
            k0, grid_size = split_volume(zero_pads((model_states['k0.k0']+model_states['k0.former_k0']).cpu(), voxel_size=voxel_size), voxel_size=voxel_size)

        print("density min", density.min())
        print("density max", density.max())
        density=density.cuda()
        k0=k0.cuda()

        big_data = torch.cat([density, k0], dim=1)
        big_data = big_data.reshape(big_data.size(0), big_data.size(1), -1)


        cnt_mask = big_data[:, 0, :]
        cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
        cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？
        if frame_id == 0:
            masks.append(torch.zeros_like(cnt_mask,device=cnt_mask.device))
            big_data_0=torch.zeros_like(big_data,device=big_data.device)
            big_data_0[:,0,:]=big_data_0[:,0,:]-4.1
            big_datas.append(big_data_0)

        masks.append(cnt_mask)

        big_datas.append(big_data)

        big_data = torch.stack(big_datas)
        residual=(big_data[1:]-big_data[:-1]) # torch.Size([frame_num, 1134, 13, 4096])

        residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)

        masks=torch.stack(masks) #mask 要改进 res为0的地方
        masks=(masks[1:] + masks[:-1]).bool()
        masks=masks.reshape((masks.size(0),-1))

        residual_cur = residual[0]
        residual_rec = torch.zeros_like(residual_cur,device=residual_cur.device)

        residual_cur=residual_cur[masks[0]]

        # bits,header=encode_jpeg(residual_cur,quality)
        bits,header=encode_jpeg_huffman(residual_cur, quality, os.path.join(path, expr_name,f'newcompressed_{frame_id}.ncrf'))
        # with open(os.path.join(path, expr_name,f'compressed_{frame_id}.ncrf'), 'wb') as compressed_file:
        #     bits.tofile(compressed_file)

        origin_size = get_origin_size(model_states['k0.k0'])
        header["origin_size"]=origin_size
        header["grid_size"]=grid_size
        header["mask_size"]=masks[0].size() #all frame same, no need to save everry time
        with open(os.path.join(path, expr_name,f'header_{frame_id}.json'), 'w') as header_f:
            json.dump(header, header_f, indent=4)

        masks_save = masks[0].bool().clone().cpu().numpy()
        masks_bit=bitarray()
        masks_bit.pack(masks_save)
        with open(os.path.join(path, expr_name, f'mask_{frame_id}.ncrf'), 'wb') as masked_file:
            masks_bit.tofile(masked_file)
        # masks_save=np.packbits(masks_save,axis=None)
        # np.save(masks_save, os.path.join(path, expr_name, f'mask_{frame_id}.npy'))

        #start = time.perf_counter()
        with open(os.path.join(path,expr_name, f'compressed_{frame_id}.ncrf'), 'rb') as compressed_file:
            with Timer("Decode",residual_cur.device):
                residual_rec_dct=decode_jpeg(compressed_file,header,residual_cur.device)

        with Timer("Recover misc", residual_cur.device):
            print("masks 0 size",masks[0].size())

            residual_rec[masks[0]]=residual_rec_dct


            residual_rec=residual_rec.reshape(residual.size(0),residual.size(1),residual.size(2),-1)
            print(residual_rec.size())
            print("grid size",grid_size)

            rec_feature=big_datas[0]+residual_rec
            rec_feature=rec_feature.squeeze(0)

            #prepare for next iteration
            cnt_mask = rec_feature[:, 0, :]
            cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
            cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？
            masks=[cnt_mask]
            big_datas = [rec_feature]

            #recover
            rec_feature = rec_feature.reshape(residual.size(1), residual.size(2), voxel_size,voxel_size,voxel_size)


            rec_feature=merge_volume(rec_feature,grid_size)

            print("origin size", origin_size)
            rec_feature=zero_unpads(rec_feature,origin_size)

        density_rec=rec_feature[:1]
        feature_rec=rec_feature[1:]

        model_states['density']=density_rec.unsqueeze(0).cpu().clone()
        model_states['k0.k0'] = feature_rec.unsqueeze(0).cpu().clone()
        model_states['k0.former_k0']=torch.zeros_like(model_states['k0.former_k0'])

        ckpt['model_state_dict']=model_states
        ckpt_path = os.path.join(path,expr_name, f'rec_{frame_id}.tar')

        torch.save(ckpt,ckpt_path)



