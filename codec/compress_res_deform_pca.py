import sys
sys.path.append('/data/new_disk2/wangla/tmp/NCVV')
import torch
import os
import copy
import numpy as np

from codec import split_volume,zero_pads,zero_unpads,get_origin_size,get_color,merge_volume,deform_warp,grid_sampler
from codec import encode_jpeg,decode_jpeg,Timer,encode_motion,decode_motion,get_masks,get_pca,encode_pca,project_pca
import time
from bitarray import bitarray
import json
import argparse

# init enviroment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, default='/data/new_disk3/wangla/tmp/NCVV/logs/NHR/jywq_sfs',
                        help='trained model path')
    parser.add_argument("--quality", type=int, default=99,
                        help='Quality of compression')
    parser.add_argument("--frame_num", type=int, default=20000,
                        help='frame_num')

    parser.add_argument("--pca", type=int, default=6)

    return parser


parser = config_parser()
args = parser.parse_args()
path = args.path

res_mode='sub'#sub or diret_save
deform_low_reso=True
pca=True

qualitys=[args.quality]
for quality in qualitys:
    expr_name = f'dynamic+{quality}_pca'
    if res_mode=='direct_save':
        expr_name+=('_'+res_mode)
    os.makedirs(os.path.join(path,expr_name),exist_ok=True)

    frame_length=args.frame_num
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
            time.sleep(1000)

        ckpt = torch.load(ckpt_path, map_location=device)
        model_states = ckpt['model_state_dict']

        if frame_id == 0:
            with open(os.path.join(path, expr_name, f'model_kwargs.json'), 'w') as header_f:
                model_kwargs=copy.deepcopy(ckpt['model_kwargs'])
                model_kwargs['xyz_min'] = model_kwargs['xyz_min'].tolist()
                model_kwargs['xyz_max'] = model_kwargs['xyz_max'].tolist()
                model_kwargs['voxel_size_ratio'] = model_kwargs['voxel_size_ratio'].tolist()
                model_kwargs['use_res'] = False
                json.dump(model_kwargs, header_f, indent=4)
            residual_k0= model_states['k0.k0']
            residual_density=model_states['density']+4.1
            masks=[torch.zeros_like(residual_density,device=residual_k0.device)-4.1,model_states['density']]
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

            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(xyz_min[0], xyz_max[0], deformation_field.shape[2]),
                torch.linspace(xyz_min[1], xyz_max[1], deformation_field.shape[3]),
                torch.linspace(xyz_min[2], xyz_max[2], deformation_field.shape[4]),
            ), -1)
            deform_xyz = deform_warp(self_grid_xyz, deformation_field,xyz_min,xyz_max)
            former_rec = grid_sampler(deform_xyz,xyz_min,xyz_max, former_rec).permute(3, 0, 1, 2).unsqueeze(0)
            if res_mode=="sub":
                residual_k0=model_states['k0.k0'] + model_states['k0.former_k0']-former_rec[:,1:]
            else:
                residual_k0=model_states['k0.former_k0']

            residual_density=model_states['density']-former_rec[:,:1]
            masks.append(model_states['density'])

        residual_full = torch.cat([residual_density, residual_k0], dim=1)
        masks=torch.cat(masks, dim=1)


        residual, grid_size = split_volume(zero_pads(residual_full, voxel_size=voxel_size),
                                          voxel_size=voxel_size)
        residual=residual.cuda()
        masks=get_masks(masks,voxel_size,residual)

        V,pca_V=get_pca(residual,masks,voxel_size)

        np.save(os.path.join(path, expr_name, 'pca_m_%d.npy' % frame_id),pca_V)
        #print(torch.matmul(V,V.transpose(0,1)))

        residual_pca=project_pca(residual_full,V)

        encode_pca(residual_pca[:,:5],masks,quality,path,expr_name,frame_id,1.)

        masks_save = masks.bool().clone().cpu().numpy()
        masks_bit=bitarray()
        masks_bit.pack(masks_save)
        with open(os.path.join(path, expr_name, f'mask_{frame_id}.ncrf'), 'wb') as masked_file:
            masks_bit.tofile(masked_file)
        # masks_save=np.packbits(masks_save,axis=None)
        # np.save(masks_save, os.path.join(path, expr_name, f'mask_{frame_id}.npy'))
        if frame_id != 0 and deform_low_reso:
            deform_save = deform_cube[masks] #??????mask??????????????????
            deform_save = deform_save.cpu().numpy()
            np.save(os.path.join(path, expr_name, 'deform_%d.npy' % frame_id), deform_save)


        #start = time.perf_counter()
        with open(os.path.join(path,expr_name, f'compressed_{frame_id}.ncrf'), 'rb') as compressed_file:
            with Timer("Decode",device):
                residual_rec_dct=decode_jpeg(compressed_file,header,residual_cur.device)

        with Timer("Recover misc", residual_cur.device):
            print("masks 0 size",masks.size())
            #inverse pca

            residual_rec[masks]=residual_rec_dct

            residual_rec=residual_rec.reshape(residual.size(0),residual.size(1),-1)
            print(residual_rec.size())
            print("grid size",grid_size)

            # rec_feature=big_datas[0]+residual_rec
            # rec_feature=rec_feature.squeeze(0)

            #recover
            rec_feature = residual_rec.reshape(residual.size(0), residual.size(1), voxel_size,voxel_size,voxel_size)

            rec_feature=merge_volume(rec_feature,grid_size)

            print("origin size", origin_size)
            rec_feature=zero_unpads(rec_feature,origin_size)

        if frame_id == 0:
            density_rec=(rec_feature[:1]-4.1).unsqueeze(0)
            feature_rec = rec_feature[1:].unsqueeze(0)
            former_rec=torch.cat([density_rec,feature_rec],dim=1)

        else:
            former_rec=rec_feature+former_rec
            density_rec=former_rec[:,:1]
            feature_rec=former_rec[:,1:]

        masks = [density_rec]

        model_states['density']=density_rec.cpu().clone()
        model_states['k0.k0'] = feature_rec.cpu().clone()
        if 'k0.former_k0' in model_states:
            model_states['k0.former_k0']=None

        ckpt['model_state_dict']=model_states
        ckpt_path = os.path.join(path,expr_name, f'rec_{frame_id}.tar')

        torch.save(ckpt,ckpt_path)



