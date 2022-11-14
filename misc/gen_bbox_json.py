import numpy as np
import os
import json
import glob

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

root_path = '/data/new_disk2/wangla/Dataset/dome/test'
log_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/test/bbox.json'

# root_path = '/data/new_disk2/wangla/Dataset/NeuralHuman/Minner_wla'
# log_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/minner/bbox.json'

os.makedirs(os.path.dirname(log_path),exist_ok=True)

camposes = np.loadtxt(os.path.join(root_path,'CamPose.inf'))
Ts = campose_to_extrinsic(camposes)
Ks = read_intrinsics(os.path.join(root_path,'Intrinsic.inf'))


# normalize to zeros , range: [-2,2]
m = np.mean(Ts[:,:3,3],axis = 0)
print('OBJ center:',m)
Ts[:,:3,3] = Ts[:,:3,3] - m
print(Ts[:,:3,3].max(),-Ts[:,:3,3].min())
#Ts[:,:3,3] = Ts[:,:3,3]*2.0/max(Ts[:,:3,3].max(),-Ts[:,:3,3].min())

min_xyzs=[]
max_xyzs=[]
num_npy=len(glob.glob(os.path.join(root_path,'pointclouds/*.npy')))
num_npy=len(glob.glob(os.path.join(root_path,'pointclouds/*.xyz')))
for i in range(1,2):
    # pts=np.load(os.path.join(root_path,f'pointclouds/frame{i}.npy'))[:,:3]
    pts=np.loadtxt(os.path.join(root_path,f'pointclouds/frame{i}.xyz'))[:,:3]
    minxyz = np.min(pts, axis=0)
    maxxyz = np.max(pts, axis=0)
    tmp=maxxyz-minxyz
    minxyz-=tmp*0.1
    maxxyz+=tmp*0.1

    min_xyzs.append(minxyz)
    max_xyzs.append(maxxyz)

min_xyzs=np.array(min_xyzs)
max_xyzs=np.array(max_xyzs)

minxyz = np.min(min_xyzs, axis=0)
maxxyz = np.max(max_xyzs, axis=0)

# minxyz=np.array( [
#         -0.22552166169126353,
#         -0.22223558800337087,
#         0.012480280346403848
#     ])
# maxxyz= np.array([
#         0.2244783383087365,
#         0.22053358510473692,
#         0.2242631728056194
#     ])
#
# tmp=maxxyz-minxyz
# minxyz-=tmp*0.3
# maxxyz+=tmp*0.3
xyzs=np.array([minxyz,maxxyz])

print('xyzs min max',xyzs)
xyzs=xyzs-m
xyzs=xyzs*2.0/max(Ts[:,:3,3].max(),-Ts[:,:3,3].min())

with open(log_path, 'w', encoding='utf-8') as f:
    json.dump({"xyz_min": xyzs[0].tolist(),
               "xyz_max": xyzs[1].tolist()}, f, ensure_ascii=False, indent=4)
#print(min_xyzs.shape)