import torch
import os
import numpy as np
from codec import psnr
import cv2

gt_path='/data/new_disk2/wangla/Projects/7.29jpeg_expr/lena.bmp'

comp_path='/data/new_disk2/wangla/Projects/7.29jpeg_expr/lena80.jpg'

psnrs=[]


img_gt=cv2.imread(gt_path)
img_comp=cv2.imread(comp_path)


psnrs.append(psnr(torch.Tensor(img_gt)/255,torch.Tensor(img_comp)/255).item())


psnrs=np.array(psnrs)

print("Mean psnr: ", psnrs.mean())