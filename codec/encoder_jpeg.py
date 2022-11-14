import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from bitarray import bitarray, bits2bytes
from prototype_jpeg.codec import DC,AC,EOB,LUMINANCE,CHROMINANCE,encode_huffman,encode_run_length,decode_huffman,\
    decode_differential,decode_run_length
from codec import dct_3d,idct_3d,quantize_quality,gen_3d_quant_tbl,Timer,merge_volume,zero_unpads
import torch.nn.functional as F
import time

from ac_dc.ncvv_ac_dc import ac_dc_encode, ac_dc_decode, ac_dc_encode2, ac_dc_decode2

QTY_3d =gen_3d_quant_tbl()

def encode_jpeg(res,quality):

    RES = dct_3d(res, norm='ortho')

    quant_table = quantize_quality(QTY_3d, quality)
    RES_quant = RES / quant_table
    data = RES_quant.cpu().numpy()
    data = np.rint(data).astype(int)
    
    #dc
    rets=[]
    data_size=data.shape
    for ch in range(data_size[1]):
        dc = data[:, ch, 0, 0, 0]
        dDC=dc[1:]-dc[:-1]
        dDC = np.concatenate((dc[:1], dDC), axis=0)

        #ret = {}
        ret_DC = ''.join(encode_huffman(v, LUMINANCE)
                          for v in dDC)

        AC=data[:, ch]

        tmp=0
        run_length_ac = []
        for block in AC:
            seq=[]
            for id in zigzag_3d:
                seq.append(block[tuple(id)])
            # tmp+=1
            # if tmp==500:
            #     print(seq)
            #     print(seq.shape)
            #     exit()
            run_length_ac.extend(
                encode_run_length(tuple(seq)[1:])
            )

        ret_AC = ''.join(encode_huffman(v, LUMINANCE)
                          for v in run_length_ac)

        rets.append(ret_DC)
        rets.append(ret_AC)

    bits = bitarray(''.join(rets))
    header={
        'size': data_size,
        'quality': quality,
        # Remaining bits length is the fake filled bits for 8 bits as a
        # byte.
        'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
        'data_slice_lengths': tuple(len(d) for d in rets)
    }
    return bits,header

def encode_jpeg_huffman(res,quality,path):

    RES = dct_3d(res, norm='ortho')

    quant_table = quantize_quality(QTY_3d, quality)
    RES_quant = RES / quant_table
    data = RES_quant.cpu().numpy()
    data = np.rint(data).astype(np.int16)
    #dc
    rets=[]
    data_size=data.shape
    ac_dc_encode2(data, 0, data_size[0], data_size[1], path)
    header={
        'size': data_size,
        'quality': quality,
        # Remaining bits length is the fake filled bits for 8 bits as a
        # byte.
      #  'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
      #  'data_slice_lengths': tuple(len(d) for d in rets)
    }
    return header

def isplit(iterable, splitter):
    ret = []
    for item in iterable:
        ret.append(item)
        if item == splitter:
            yield ret
            ret = []

def decode_jpeg(file_object,header,device):
    bits = bitarray()
    bits.fromfile(file_object)
    print("len bits", len(bits))
    bits = bits.to01()

    data_size = header['size']
    quality = header['quality']
    remaining_bits_length = header['remaining_bits_length']
    dsls = header['data_slice_lengths']  # data_slice_lengths

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure for `Decoder`.

    if remaining_bits_length:
        bits = bits[:-remaining_bits_length]

    sliced=[]
    start=0
    end=0
    for i in range(len(dsls)):
        start=end
        end+=dsls[i]
        sliced.append(bits[start:end])

    with Timer("DC AC", device):
        rec=np.zeros(data_size)
        for ch in range(data_size[1]):
            #get dc
            dcs=tuple(decode_differential(decode_huffman(sliced[2*ch],DC,LUMINANCE)))
            #rec[:,ch, 0, 0, 0]=torch.tensor(dc)

            #get ac
            acs = tuple(decode_run_length(pairs) for pairs in isplit(
                decode_huffman(sliced[2*ch+1], AC, LUMINANCE),
                EOB
            ))

            if len(dcs) != len(acs):
                raise ValueError(f'DC size {len(dcs)} is not equal to AC size '
                                 f'{len(acs)}.')
            # for ac in acs:
            #     print(len(ac))

            for ii,(dc,ac) in enumerate(zip(dcs,acs)):
                tmp=(dc, ) + ac
                block=np.zeros((8,8,8))
                for i in range(len(tmp)):
                    block[tuple(zigzag_3d[i])]=tmp[i]

                rec[ii,ch]=block

    rec = torch.tensor(rec, device=device).to(torch.float)

    quant_table = quantize_quality(QTY_3d, quality)
    rec=rec*quant_table

    with Timer("idct3d", device):
        rec = idct_3d(rec, norm='ortho')
    return rec

def decode_jpeg_huffman(file_object,header,device):
    entropy_time = 0.0
    quant_time = 0.0
    idct_time = 0.0
    data_size = header['size']
    quality = header['quality']

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure
    #for `Decoder`.

    t = time.time()
    rec=np.zeros(data_size, dtype=np.int16)
    ac_dc_decode2(rec, 0, data_size[0], data_size[1], file_object)
    rec = torch.tensor(rec, device=device).to(torch.float)
    entropy_time = time.time() - t

    t = time.time()
    quant_table = quantize_quality(QTY_3d, quality)
    rec=rec*quant_table
    quant_time = time.time() - t

    t = time.time()
    rec = idct_3d(rec, norm='ortho')
    idct_time = time.time() - t

    return rec, entropy_time, quant_time, idct_time

def encode_jpeg_huffman_cpu(res,quality,path):

    RES = dct_3d(res, norm='ortho')

    quant_table = quantize_quality(QTY_3d, quality)
    RES_quant = RES.cpu() / quant_table.cpu()
    data = RES_quant.numpy()
    data = np.rint(data).astype(np.int16)
    #dc
    rets=[]
    data_size=data.shape
    ac_dc_encode2(data, 0, data_size[0], data_size[1], path)
    header={
        'size': data_size,
        'quality': quality,
        # Remaining bits length is the fake filled bits for 8 bits as a
        # byte.
      #  'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
      #  'data_slice_lengths': tuple(len(d) for d in rets)
    }
    return header

def decode_jpeg_huffman_cpu(file_object,header,device):

    data_size = header['size']
    quality = header['quality']

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure
    #for `Decoder`.

    # with Timer("DC AC", device):
    rec=np.zeros(data_size, dtype=np.int16)
    ac_dc_decode2(rec, 0, data_size[0], data_size[1], file_object)
    
    rec = torch.tensor(rec, device=device).to(torch.float)

    quant_table = quantize_quality(QTY_3d, quality)
    rec=rec.cpu()*quant_table.cpu()

    # with Timer("idct3d", device):
    rec = idct_3d(rec, norm='ortho')
    return rec

def dc_ac(data_size,sliced):
    rec = np.zeros(data_size)
    for ch in range(data_size[1]):
        # get dc
        dcs = tuple(decode_differential(decode_huffman(sliced[2 * ch], DC, LUMINANCE)))

        # get ac
        acs = tuple(decode_run_length(pairs) for pairs in isplit(
            decode_huffman(sliced[2 * ch + 1], AC, LUMINANCE),
            EOB
        ))

        if len(dcs) != len(acs):
            raise ValueError(f'DC size {len(dcs)} is not equal to AC size '
                             f'{len(acs)}.')

        for ii, (dc, ac) in enumerate(zip(dcs, acs)):
            tmp = (dc,) + ac
            block = np.zeros((8, 8, 8))
            for i in range(len(tmp)):
                block[tuple(zigzag_3d[i])] = tmp[i]

            rec[ii, ch] = block

    return rec

@torch.no_grad()
def decode_jpeg_mmap(file_bits,header,device):
    print("len bits", len(file_bits))
    bits = file_bits.to01()

    data_size = header['size']
    quality = header['quality']
    remaining_bits_length = header['remaining_bits_length']
    dsls = header['data_slice_lengths']  # data_slice_lengths

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure for `Decoder`.

    if remaining_bits_length:
        bits = bits[:-remaining_bits_length]

    sliced=[]
    start=0
    end=0
    for i in range(len(dsls)):
        start=end
        end+=dsls[i]
        sliced.append(bits[start:end])

    print("the file length should be ", end+remaining_bits_length)

    with Timer("DC AC", device):
        rec=dc_ac(data_size,sliced)


    rec = torch.tensor(rec, device=device).to(torch.float)

    quant_table = quantize_quality(QTY_3d, quality)
    rec=rec*quant_table

    with Timer("idct3d", device):
        rec = idct_3d(rec, norm='ortho')
    return rec

@torch.no_grad()
def recover_misc(residual_rec_dct,former_rec,header, mask,n_channel=13,voxel_size=8,device='cuda'):
    with Timer("Recover misc", device):
        residual_rec=torch.zeros((mask.size(0),n_channel,voxel_size,voxel_size,voxel_size),device=device)
        residual_rec[mask] = residual_rec_dct

        residual_rec = residual_rec.reshape( mask.size(0), n_channel, -1)
        print(former_rec.size())
        print(residual_rec.size())
        rec_feature = former_rec + residual_rec

        # recover
        rec_feature = rec_feature.reshape( mask.size(0), n_channel, voxel_size, voxel_size,
                                          voxel_size)
        grid_size=header["grid_size"]
        origin_size=header["origin_size"]
        rec_feature = merge_volume(rec_feature, grid_size)
        rec_feature = zero_unpads(rec_feature, origin_size).unsqueeze(0)

    return rec_feature


@torch.no_grad()
def deform_warp(xyz,deformation_field,xyz_min,xyz_max, align_corners=True):
    '''

    :param xyz: [N,3]
    :param align_corners:
    :return:
    '''
    mode = 'bilinear'

    shape = xyz.shape[:-1]

    xyz_r = xyz.reshape(1, 1, 1, -1, 3)
    ind_norm = ((xyz_r - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1

    deform_vector = F.grid_sample(deformation_field, ind_norm, mode=mode, align_corners=align_corners). \
        reshape(deformation_field.shape[1], -1).T.reshape(*shape, deformation_field.shape[1])

    xyz = xyz + deform_vector

    return xyz

@torch.no_grad()
def grid_sampler( xyz,xyz_min,xyz_max, *grids,mode=None, align_corners=True):
    '''Wrapper for the interp operation'''
    mode = 'bilinear'
    shape = xyz.shape[:-1]

    xyz = xyz.reshape(1, 1, 1, -1, 3)
    ind_norm = ((xyz - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1

    ret_lst = [
        # TODO: use `rearrange' to make it readable
        F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(
            *shape, grid.shape[1])
        for grid in grids
    ]
    for i in range(len(grids)):
        if ret_lst[i].shape[-1] == 1:
            ret_lst[i] = ret_lst[i].squeeze(-1)
    if len(ret_lst) == 1:
        return ret_lst[0]
    return ret_lst

@torch.no_grad()
def decode_motion(deform,grid_size,origin_size,voxel_size=8):
    deform=deform.unsqueeze(-1)
    deform=deform.repeat(1,1,voxel_size**3)
    deform=deform.reshape(deform.size(0),deform.size(1),voxel_size,voxel_size,voxel_size)
    deform=merge_volume(deform,grid_size)
    deform=zero_unpads(deform,origin_size).unsqueeze(0)
    return deform

@torch.no_grad()
def recover_misc_deform(residual_rec_dct,former_rec,header, mask,deform,model_states, n_channel=13,voxel_size=8,device='cuda'):
    with Timer("Recover misc", device):
        grid_size = header["grid_size"]
        origin_size = header["origin_size"]
        deform_rec = torch.zeros((mask.size(0), 3), device=device)
        deform_rec[mask] = deform
        deformation_field = decode_motion(deform_rec, grid_size, origin_size)

        xyz_min = torch.Tensor(model_states['xyz_min'],device=device)
        xyz_max = torch.Tensor(model_states['xyz_max'],device=device)

        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(xyz_min[0], xyz_max[0], deformation_field.shape[2]),
            torch.linspace(xyz_min[1], xyz_max[1], deformation_field.shape[3]),
            torch.linspace(xyz_min[2], xyz_max[2], deformation_field.shape[4]),
        ), -1)
        deform_xyz = deform_warp(self_grid_xyz, deformation_field, xyz_min, xyz_max)
        former_rec = grid_sampler(deform_xyz, xyz_min, xyz_max, former_rec).permute(3, 0, 1, 2).unsqueeze(0)

        residual_rec = torch.zeros((mask.size(0), n_channel, voxel_size, voxel_size, voxel_size), device=device)
        residual_rec[mask] = residual_rec_dct

        residual_rec = residual_rec.reshape(mask.size(0), n_channel, -1)
#        print(former_rec.size())
#        print(residual_rec.size())
        #rec_feature = former_rec + residual_rec
        # recover
        rec_feature = residual_rec.reshape(mask.size(0), n_channel, voxel_size, voxel_size,
                                          voxel_size)
        grid_size = header["grid_size"]
        origin_size = header["origin_size"]
        rec_feature = merge_volume(rec_feature, grid_size)
        rec_feature = zero_unpads(rec_feature, origin_size).unsqueeze(0)

        rec_feature=former_rec+rec_feature

    return rec_feature

# # translate from https://github.com/balcilar/3dFDCT/blob/7bcb253a1513afb19b388f1e6b463dd48a41573f/zigzag3d.m

# 可能matlab是优先纵轴，python优先横轴，但感觉影响不大 （后面两个维度？）
zigzag_3d=[[0,0,0],
[0,0,1],
[0,1,0],
[1,0,0],
[2,0,0],
[1,1,0],
[1,0,1],
[0,2,0],
[0,1,1],
[0,0,2],
[0,0,3],
[0,1,2],
[0,2,1],
[0,3,0],
[1,0,2],
[1,1,1],
[1,2,0],
[2,0,1],
[2,1,0],
[3,0,0],
[4,0,0],
[3,1,0],
[3,0,1],
[2,2,0],
[2,1,1],
[2,0,2],
[1,3,0],
[1,2,1],
[1,1,2],
[1,0,3],
[0,4,0],
[0,3,1],
[0,2,2],
[0,1,3],
[0,0,4],
[0,0,5],
[0,1,4],
[0,2,3],
[0,3,2],
[0,4,1],
[0,5,0],
[1,0,4],
[1,1,3],
[1,2,2],
[1,3,1],
[1,4,0],
[2,0,3],
[2,1,2],
[2,2,1],
[2,3,0],
[3,0,2],
[3,1,1],
[3,2,0],
[4,0,1],
[4,1,0],
[5,0,0],
[6,0,0],
[5,1,0],
[5,0,1],
[4,2,0],
[4,1,1],
[4,0,2],
[3,3,0],
[3,2,1],
[3,1,2],
[3,0,3],
[2,4,0],
[2,3,1],
[2,2,2],
[2,1,3],
[2,0,4],
[1,5,0],
[1,4,1],
[1,3,2],
[1,2,3],
[1,1,4],
[1,0,5],
[0,6,0],
[0,5,1],
[0,4,2],
[0,3,3],
[0,2,4],
[0,1,5],
[0,0,6],
[0,0,7],
[0,1,6],
[0,2,5],
[0,3,4],
[0,4,3],
[0,5,2],
[0,6,1],
[0,7,0],
[1,0,6],
[1,1,5],
[1,2,4],
[1,3,3],
[1,4,2],
[1,5,1],
[1,6,0],
[2,0,5],
[2,1,4],
[2,2,3],
[2,3,2],
[2,4,1],
[2,5,0],
[3,0,4],
[3,1,3],
[3,2,2],
[3,3,1],
[3,4,0],
[4,0,3],
[4,1,2],
[4,2,1],
[4,3,0],
[5,0,2],
[5,1,1],
[5,2,0],
[6,0,1],
[6,1,0],
[7,0,0],
[7,1,0],
[7,0,1],
[6,2,0],
[6,1,1],
[6,0,2],
[5,3,0],
[5,2,1],
[5,1,2],
[5,0,3],
[4,4,0],
[4,3,1],
[4,2,2],
[4,1,3],
[4,0,4],
[3,5,0],
[3,4,1],
[3,3,2],
[3,2,3],
[3,1,4],
[3,0,5],
[2,6,0],
[2,5,1],
[2,4,2],
[2,3,3],
[2,2,4],
[2,1,5],
[2,0,6],
[1,7,0],
[1,6,1],
[1,5,2],
[1,4,3],
[1,3,4],
[1,2,5],
[1,1,6],
[1,0,7],
[0,7,1],
[0,6,2],
[0,5,3],
[0,4,4],
[0,3,5],
[0,2,6],
[0,1,7],
[0,2,7],
[0,3,6],
[0,4,5],
[0,5,4],
[0,6,3],
[0,7,2],
[1,1,7],
[1,2,6],
[1,3,5],
[1,4,4],
[1,5,3],
[1,6,2],
[1,7,1],
[2,0,7],
[2,1,6],
[2,2,5],
[2,3,4],
[2,4,3],
[2,5,2],
[2,6,1],
[2,7,0],
[3,0,6],
[3,1,5],
[3,2,4],
[3,3,3],
[3,4,2],
[3,5,1],
[3,6,0],
[4,0,5],
[4,1,4],
[4,2,3],
[4,3,2],
[4,4,1],
[4,5,0],
[5,0,4],
[5,1,3],
[5,2,2],
[5,3,1],
[5,4,0],
[6,0,3],
[6,1,2],
[6,2,1],
[6,3,0],
[7,0,2],
[7,1,1],
[7,2,0],
[7,3,0],
[7,2,1],
[7,1,2],
[7,0,3],
[6,4,0],
[6,3,1],
[6,2,2],
[6,1,3],
[6,0,4],
[5,5,0],
[5,4,1],
[5,3,2],
[5,2,3],
[5,1,4],
[5,0,5],
[4,6,0],
[4,5,1],
[4,4,2],
[4,3,3],
[4,2,4],
[4,1,5],
[4,0,6],
[3,7,0],
[3,6,1],
[3,5,2],
[3,4,3],
[3,3,4],
[3,2,5],
[3,1,6],
[3,0,7],
[2,7,1],
[2,6,2],
[2,5,3],
[2,4,4],
[2,3,5],
[2,2,6],
[2,1,7],
[1,7,2],
[1,6,3],
[1,5,4],
[1,4,5],
[1,3,6],
[1,2,7],
[0,7,3],
[0,6,4],
[0,5,5],
[0,4,6],
[0,3,7],
[0,4,7],
[0,5,6],
[0,6,5],
[0,7,4],
[1,3,7],
[1,4,6],
[1,5,5],
[1,6,4],
[1,7,3],
[2,2,7],
[2,3,6],
[2,4,5],
[2,5,4],
[2,6,3],
[2,7,2],
[3,1,7],
[3,2,6],
[3,3,5],
[3,4,4],
[3,5,3],
[3,6,2],
[3,7,1],
[4,0,7],
[4,1,6],
[4,2,5],
[4,3,4],
[4,4,3],
[4,5,2],
[4,6,1],
[4,7,0],
[5,0,6],
[5,1,5],
[5,2,4],
[5,3,3],
[5,4,2],
[5,5,1],
[5,6,0],
[6,0,5],
[6,1,4],
[6,2,3],
[6,3,2],
[6,4,1],
[6,5,0],
[7,0,4],
[7,1,3],
[7,2,2],
[7,3,1],
[7,4,0],
[7,5,0],
[7,4,1],
[7,3,2],
[7,2,3],
[7,1,4],
[7,0,5],
[6,6,0],
[6,5,1],
[6,4,2],
[6,3,3],
[6,2,4],
[6,1,5],
[6,0,6],
[5,7,0],
[5,6,1],
[5,5,2],
[5,4,3],
[5,3,4],
[5,2,5],
[5,1,6],
[5,0,7],
[4,7,1],
[4,6,2],
[4,5,3],
[4,4,4],
[4,3,5],
[4,2,6],
[4,1,7],
[3,7,2],
[3,6,3],
[3,5,4],
[3,4,5],
[3,3,6],
[3,2,7],
[2,7,3],
[2,6,4],
[2,5,5],
[2,4,6],
[2,3,7],
[1,7,4],
[1,6,5],
[1,5,6],
[1,4,7],
[0,7,5],
[0,6,6],
[0,5,7],
[0,6,7],
[0,7,6],
[1,5,7],
[1,6,6],
[1,7,5],
[2,4,7],
[2,5,6],
[2,6,5],
[2,7,4],
[3,3,7],
[3,4,6],
[3,5,5],
[3,6,4],
[3,7,3],
[4,2,7],
[4,3,6],
[4,4,5],
[4,5,4],
[4,6,3],
[4,7,2],
[5,1,7],
[5,2,6],
[5,3,5],
[5,4,4],
[5,5,3],
[5,6,2],
[5,7,1],
[6,0,7],
[6,1,6],
[6,2,5],
[6,3,4],
[6,4,3],
[6,5,2],
[6,6,1],
[6,7,0],
[7,0,6],
[7,1,5],
[7,2,4],
[7,3,3],
[7,4,2],
[7,5,1],
[7,6,0],
[7,7,0],
[7,6,1],
[7,5,2],
[7,4,3],
[7,3,4],
[7,2,5],
[7,1,6],
[7,0,7],
[6,7,1],
[6,6,2],
[6,5,3],
[6,4,4],
[6,3,5],
[6,2,6],
[6,1,7],
[5,7,2],
[5,6,3],
[5,5,4],
[5,4,5],
[5,3,6],
[5,2,7],
[4,7,3],
[4,6,4],
[4,5,5],
[4,4,6],
[4,3,7],
[3,7,4],
[3,6,5],
[3,5,6],
[3,4,7],
[2,7,5],
[2,6,6],
[2,5,7],
[1,7,6],
[1,6,7],
[0,7,7],
[1,7,7],
[2,6,7],
[2,7,6],
[3,5,7],
[3,6,6],
[3,7,5],
[4,4,7],
[4,5,6],
[4,6,5],
[4,7,4],
[5,3,7],
[5,4,6],
[5,5,5],
[5,6,4],
[5,7,3],
[6,2,7],
[6,3,6],
[6,4,5],
[6,5,4],
[6,6,3],
[6,7,2],
[7,1,7],
[7,2,6],
[7,3,5],
[7,4,4],
[7,5,3],
[7,6,2],
[7,7,1],
[7,7,2],
[7,6,3],
[7,5,4],
[7,4,5],
[7,3,6],
[7,2,7],
[6,7,3],
[6,6,4],
[6,5,5],
[6,4,6],
[6,3,7],
[5,7,4],
[5,6,5],
[5,5,6],
[5,4,7],
[4,7,5],
[4,6,6],
[4,5,7],
[3,7,6],
[3,6,7],
[2,7,7],
[3,7,7],
[4,6,7],
[4,7,6],
[5,5,7],
[5,6,6],
[5,7,5],
[6,4,7],
[6,5,6],
[6,6,5],
[6,7,4],
[7,3,7],
[7,4,6],
[7,5,5],
[7,6,4],
[7,7,3],
[7,7,4],
[7,6,5],
[7,5,6],
[7,4,7],
[6,7,5],
[6,6,6],
[6,5,7],
[5,7,6],
[5,6,7],
[4,7,7],
[5,7,7],
[6,6,7],
[6,7,6],
[7,5,7],
[7,6,6],
[7,7,5],
[7,7,6],
[7,6,7],
[6,7,7],
[7,7,7]]
