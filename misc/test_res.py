import torch
import os
import copy
import numpy as np

path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_res'

ckpt_path = os.path.join(path, 'fine_last_%d.tar' % 0)

ckpt = torch.load(ckpt_path, map_location='cpu')

k0=ckpt['model_state_dict']['k0.k0']

ckpt_path = os.path.join(path, 'fine_last_%d.tar' % 1)

ckpt = torch.load(ckpt_path, map_location='cpu')

print(torch.all(k0==ckpt['model_state_dict']['k0.former_k0']))