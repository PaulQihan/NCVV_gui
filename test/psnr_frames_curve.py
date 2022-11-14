import matplotlib.pyplot as plt
import numpy as np
from codec import psnr
import cv2
import torch

folder1='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/render_36_fine_last_%d_%d'
folder2='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/render_36_rec_%d_%d'
frame_num=600

psnrs=[]
frame_ids=[]
for i in range(1,frame_num):
    psnr_single=[]
    for j in range(10):
        path1=f'/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/render_36_fine_last_{i}_{i}/'+str(j).zfill(3)+'.jpg'
        path2=f'/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/render_36_rec_{i}_{i}/'+str(j).zfill(3)+'.jpg'

        img_gt = cv2.imread(path1)
        img_comp = cv2.imread(path2)

        psnr_single.append(psnr(torch.Tensor(img_gt) / 255, torch.Tensor(img_comp) / 255).item())

    psnr_mean=np.array(psnr_single).mean()
    psnrs.append(psnr_mean)
    with open(f"/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/render_36_rec_{i}_{i}/psnr.txt","w") as f:
        f.write('%f\n' % float(psnr_mean))

    frame_ids.append(i)
    print("finish frame ", i)

psnr=np.asarray(psnrs)
frame_ids=np.asarray(frame_ids)
plt.xlabel("Frame Num")
plt.ylabel("PSNR")
plt.ylim(35,55)
plt.plot(frame_ids,psnr)
plt.savefig(f"/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_new_mask1024/psnr_frames_curve_{frame_num}.png")
plt.close()