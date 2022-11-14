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
import argparse

def get_masks(masks,voxel_size,residual):
    masks, grid_size_mask = split_volume(zero_pads(masks, voxel_size=voxel_size),
                                         voxel_size=voxel_size)
    masks = masks.cuda()

    masks = masks.reshape(residual.size(0), 2, -1)
    masks = torch.nn.functional.softplus(masks - 4.1) > 0.4
    masks = masks.sum(dim=-1)  # 用来mask全0的voxel？minye的经验值，之后的数据可能需要细调？？？

    # residual=residual.reshape(residual.size(0),residual.size(1),residual.size(2),voxel_size,voxel_size,voxel_size)

    masks = (masks[:, 0] + masks[:, 1]).bool()

    return masks

def get_pca(residual,masks,voxel_size):
    residual_cur = residual[masks]
    # start pca
    tmp = residual_cur.reshape(residual_cur.size(0), residual_cur.size(1), -1)
    tmp = tmp.permute((0, 2, 1))  #
    tmp = tmp.reshape(-1, tmp.size(-1))[:, 1:]  # [N,12]
    # print(tmp.size())
    _, _, V = torch.pca_lowrank(tmp, q=tmp.size(-1))
    print("V", V.size())
    # residual_pca = torch.matmul(tmp, V)
    # V6=V[:,:12]
    # print(torch.dist(tmp@V6 @(V6.transpose(0,1)), tmp))
    # residual_pca = residual_pca.reshape(residual_cur.size(0), voxel_size ** 3, tmp.size(-1))
    # residual_pca = residual_pca.permute((0, 2, 1))
    # residual_pca = residual_pca.reshape(residual_pca.size(0), residual_pca.size(1), voxel_size, voxel_size, voxel_size)
    # residual_cur = torch.cat([residual_cur[:, :1], residual_pca[:, :args.pca]], dim=1)

    pca_V = V.transpose(0, 1).cpu().numpy()

    return V,pca_V

def project_pca(residual_full,V):
    '''

    :param residual_full: [1, 13, 147, 322, 122]
    :param V:
    :return:
    '''
    residual_pca=residual_full[0,1:]
    residual_pca=residual_pca.reshape(residual_pca.size(0),-1).transpose(0,1)

    residual_pca=residual_pca@V
    residual_pca=residual_pca.transpose(0,1).reshape(1,residual_pca.size(0),
                                 residual_full.size(2),residual_full.size(3),residual_full.size(4))
    return torch.cat([residual_full[:,:1],residual_pca],dim=1)


def encode_pca(residual,masks,quality,path,expr_name,frame_id,reso):
    if reso!=1:
        pass


    bits, header = encode_jpeg(residual_cur, quality)

    with open(os.path.join(path, expr_name, f'compressed_{frame_id}.ncrf'), 'wb') as compressed_file:
        bits.tofile(compressed_file)

    origin_size = get_origin_size(model_states['k0.k0'])
    header["origin_size"] = origin_size
    header["grid_size"] = grid_size
    header["mask_size"] = masks.size()  # all frame same, no need to save everry time
    with open(os.path.join(path, expr_name, f'header_{frame_id}.json'), 'w') as header_f:
        json.dump(header, header_f, indent=4)