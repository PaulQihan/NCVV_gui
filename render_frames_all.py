import torch
import os
import numpy as np
from codec import psnr
import cv2
import subprocess

frame_num=600
for i in range(1,frame_num):
    cmd = f'python run.py --config configs/NHR/dance_new.py --render_only --render_360 {i} --render_360_step 36'
    cmd = f'python run.py --config configs/NHR/dance_new.py --render_only --render_360 {i} --render_360_step 36 --ckpt_name dynamic+99/rec_%d.tar'
    cmd = cmd.split()
    process = subprocess.Popen(cmd)
    process.wait()