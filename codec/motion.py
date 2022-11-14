import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from bitarray import bitarray, bits2bytes
import torch.nn.functional as F
from codec import split_volume,zero_pads,zero_unpads,get_origin_size,Timer,merge_volume,zero_unpads

@torch.no_grad()
def encode_motion(deformation_field,voxel_size=8):
    #mode cube
    origin_size = get_origin_size(deformation_field)
    deform, grid_size = split_volume(zero_pads(deformation_field, voxel_size=voxel_size),
                                      voxel_size=voxel_size)

    deform=deform.reshape(deform.size(0),deform.size(1),-1)
    deform=torch.mean(deform,dim=-1)

    return deform, grid_size,origin_size

