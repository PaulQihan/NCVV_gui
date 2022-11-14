import torch
import os
import numpy as np
from codec import psnr
import cv2
import subprocess

qualities = (99,100,98,96,94,92,90)

# 'fine_last_{frame_id}_recq_{q}_quality.tar'
# cmd=f'python run.py --config configs/NHR/xzq_wmyparams_res_l1.py --render_only --render_360 1'
# cmd=cmd.split()
# process = subprocess.Popen(cmd)
# process.wait()
for q in qualities:



    cmd = f"python run.py --config configs/NHR/xzq_wmyparams.py --render_only --render_train 1  --ckpt_name dynamic+{q}/rec_%d.tar"
    cmd = cmd.split()
    process = subprocess.Popen(cmd)
    process.wait()

    # path_before="/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-1/render_1__1"
    #
    # path_after=f"/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-1/render_1_dynamic+{q}/rec_%d.tar_1"
    #
    # print(path_after)
    # psnrs=[]
    #
    # for i in range(360):
    #     img_gt=cv2.imread(os.path.join(path_before, str(i).zfill(3)+'.jpg'))
    #     img_comp=cv2.imread(os.path.join(path_after, str(i).zfill(3)+'.jpg'))
    #
    #     psnrs.append(psnr(torch.Tensor(img_gt)/255,torch.Tensor(img_comp)/255).item())
    #
    #
    # psnrs=np.array(psnrs)
    #
    # print("print", q, "Mean psnr: ", psnrs.mean())
    # with open(f"/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-1/render_1_dynamic+{q}/psnr_{q}.txt","w") as f:
    #     f.write('%f\n' % float(psnrs.mean()))


def test2():

    path_before="/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2/render_360_dyanframes_fine_last_0_19"

    path_after=f"/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2/render_360_dyanframes_rec_0_10"

    psnrs=[]

    for i in range(9):
        img_gt=cv2.imread(os.path.join(path_before, str(i).zfill(3)+'.jpg'))
        img_comp=cv2.imread(os.path.join(path_after, str(i).zfill(3)+'.jpg'))


        psnrs.append(psnr(torch.Tensor(img_gt)/255,torch.Tensor(img_comp)/255).item())


    psnrs=np.array(psnrs)

    print("print",  "Mean psnr: ", psnrs.mean())
