import sys
sys.path.append('/data/new_disk2/wangla/tmp/NCVV')
import torch
import os
import copy
import numpy as np

from codec import split_volume,zero_pads,zero_unpads,get_origin_size,get_color,merge_volume,deform_warp,grid_sampler
from codec import encode_jpeg,decode_jpeg,Timer,encode_motion,decode_motion
import time
from bitarray import bitarray
import json

# init enviroment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_deform_tv_res_cube_d_l1'

res_mode='sub'#sub or diret_save
deform_low_reso=True

qualitys=[99]
for quality in qualitys:
    expr_name = f'dynamic+{quality}_old'
    if res_mode=='direct_save':
        expr_name+=('_'+res_mode)
    os.makedirs(os.path.join(path,expr_name),exist_ok=True)

    frame_length=200
    voxel_size = 8

    masks=[]


    thresh = 1

    quant_type="jpeg"

    for frame_id in range(frame_length):
        print('process frame', frame_id)
        ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
        while not os.path.exists(ckpt_path):
            torch.cuda.empty_cache()
            print("waiting checkpoint ", ckpt_path)
            time.sleep(3000)

        ckpt = torch.load(ckpt_path, map_location=device)
        model_states = ckpt['model_state_dict']

        if frame_id == 0:
            residual_k0= model_states['k0.k0']
            residual_density=model_states['density']+4.1
        else:

                # for deformation field
            ckpt_deform_path = os.path.join(path, 'fine_last_%d_deform.tar' % frame_id)
            ckpt_deform = torch.load(ckpt_deform_path, map_location=device)
            model_states_deform = ckpt_deform['model_state_dict']
            deformation_field = model_states_deform['deformation_field']
            if deform_low_reso:
                deform_cube, grid_size, origin_size= encode_motion(deformation_field)
                deformation_field = decode_motion(deform_cube, grid_size, origin_size)

            xyz_min=model_states['xyz_min']
            xyz_max=model_states['xyz_max']
            former_k0=feature_rec
            # print(former_k0.size())
            # print(model_states['k0.former_k0'].size())
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(xyz_min[0], xyz_max[0], deformation_field.shape[2]),
                torch.linspace(xyz_min[1], xyz_max[1], deformation_field.shape[3]),
                torch.linspace(xyz_min[2], xyz_max[2], deformation_field.shape[4]),
            ), -1)
            deform_xyz = deform_warp(self_grid_xyz, deformation_field,xyz_min,xyz_max)
            former_k0 = grid_sampler(deform_xyz,xyz_min,xyz_max, former_k0).permute(3, 0, 1, 2).unsqueeze(0)
            if res_mode=="sub":
                residual_k0=model_states['k0.k0'] + model_states['k0.former_k0']-former_k0
            else:
                residual_k0=model_states['k0.former_k0']

            residual_density=model_states['density']-density_rec

        residual = torch.cat([residual_density, residual_k0], dim=1)


        residual, grid_size = split_volume(zero_pads(residual, voxel_size=voxel_size),
                                          voxel_size=voxel_size)
        residual=residual.cuda()

        cnt_mask = residual[:, 0, :].reshape(residual.size(0),-1)
        cnt_mask = torch.nn.functional.softplus(cnt_mask - 4.1) > 0.4
        cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？
        if frame_id == 0:
            masks.append(torch.zeros_like(cnt_mask,device=cnt_mask.device))
            former_density_cube = torch.zeros_like(residual[:, 0, :].reshape(residual.size(0),-1), device=residual.device)
            former_density_cube -= 4.1

        masks.append(cnt_mask)
        print(residual.size())

        #residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)

        masks=torch.stack(masks)
        masks=(masks[1:] + masks[:-1]).bool()
        masks=masks.reshape((masks.size(0),-1))

        residual_rec = torch.zeros_like(residual,device=residual.device)

        residual_cur=residual[masks[0]]

        bits,header=encode_jpeg(residual_cur,quality)

        with open(os.path.join(path, expr_name,f'compressed_{frame_id}.ncrf'), 'wb') as compressed_file:
            bits.tofile(compressed_file)

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
        if frame_id != 0 and deform_low_reso:
            deform_save = deform_cube[masks[0]] #这个mask的计算也要变
            deform_save = deform_save.cpu().numpy()
            np.save(os.path.join(path, expr_name, 'deform_%d.npy' % frame_id), deform_save)


        #start = time.perf_counter()
        with open(os.path.join(path,expr_name, f'compressed_{frame_id}.ncrf'), 'rb') as compressed_file:
            with Timer("Decode",residual_cur.device):
                residual_rec_dct=decode_jpeg(compressed_file,header,residual_cur.device)

        with Timer("Recover misc", residual_cur.device):
            print("masks 0 size",masks[0].size())

            residual_rec[masks[0]]=residual_rec_dct

            residual_rec=residual_rec.reshape(residual.size(0),residual.size(1),-1)
            print(residual_rec.size())
            print("grid size",grid_size)

            # rec_feature=big_datas[0]+residual_rec
            # rec_feature=rec_feature.squeeze(0)
            former_density_cube+=residual_rec[:,0,:]

            #prepare for next iteration
            cnt_mask = torch.nn.functional.softplus(former_density_cube - 4.1) > 0.4
            cnt_mask = cnt_mask.sum(dim=1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？
            masks=[cnt_mask]

            #recover
            rec_feature = residual_rec.reshape(residual.size(0), residual.size(1), voxel_size,voxel_size,voxel_size)

            rec_feature=merge_volume(rec_feature,grid_size)

            print("origin size", origin_size)
            rec_feature=zero_unpads(rec_feature,origin_size)

        if frame_id == 0:
            density_rec=rec_feature[:1]-4.1
            feature_rec = rec_feature[1:].unsqueeze(0)
        else:
            density_rec=rec_feature[:1]+density_rec
            feature_rec=rec_feature[1:]+former_k0

        model_states['density']=density_rec.unsqueeze(0).cpu().clone()
        model_states['k0.k0'] = feature_rec.cpu().clone()
        if 'k0.former_k0' in model_states:
            model_states['k0.former_k0']=None

        ckpt['model_state_dict']=model_states
        ckpt_path = os.path.join(path,expr_name, f'rec_{frame_id}.tar')

        torch.save(ckpt,ckpt_path)



