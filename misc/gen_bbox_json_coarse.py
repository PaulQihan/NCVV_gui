import numpy as np
import os
import json
import glob
import torch

def campose_to_extrinsic(camposes):
    if camposes.shape[1] != 12:
        raise Exception(" wrong campose data structure!")
        return

    res = np.zeros((camposes.shape[0], 4, 4))

    res[:, 0:3, 2] = camposes[:, 0:3]
    res[:, 0:3, 0] = camposes[:, 3:6]
    res[:, 0:3, 1] = camposes[:, 6:9]
    res[:, 0:3, 3] = camposes[:, 9:12]
    res[:, 3, 3] = 1.0

    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data = fo.readlines()
    i = 0
    Ks = []
    while i < len(data):
        if len(data[i]) > 5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i + 1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i + 1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a, b, c])
            Ks.append(res)

        i = i + 1
    Ks = np.stack(Ks)
    fo.close()

    return Ks

root_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/test'
log_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/test/bbox.json'

# root_path = '/data/new_disk2/wangla/Dataset/NeuralHuman/Minner_wla'
# log_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/minner/bbox.json'

os.makedirs(os.path.dirname(log_path),exist_ok=True)

ckpt_path = os.path.join(root_path, 'coarse_last_%d.tar' % 0)

ckpt = torch.load(ckpt_path, map_location='cpu')
model_states = ckpt['model_state_dict']
print(model_states['xyz_min'],model_states['xyz_max'])