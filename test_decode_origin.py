
'''
要传的数据：
新场景一开始需要传的：model_kwargs.json，rgbnet.tar
每一帧需要传的:header_{frame_count}.json,mask_{frame_count}.ncrf,deform_{frame_count}.npy(第0帧没有),compressed.nrcf.(可能还有pca_%d.npy,还没实现)
'''

import argparse
import mmap
import numpy as np
import time
import sys
import json
import os
from bitarray import bitarray
from codec import  decode_jpeg_mmap,Timer,recover_misc,recover_misc_deform,decode_jpeg_huffman
import torch

from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import gc
import ipdb

from lib import utils, dvgo, dmpigo, dvgo_video
from lib.load_data import load_data, load_data_frame
from tools.voxelized import sample_grid_on_voxel
from run import render_viewpoints,seed_everything,load_everything_frame
from render_dyna import render_viewpoints_frames, rodrigues_rotation_matrix,read_intrinsics,campose_to_extrinsic

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')

    parser.add_argument("--mmap_path", type=str, default='',
                        help='mmap file path')

    parser.add_argument("--model_path", type=str, default='',
                        help='mask file path')

    parser.add_argument("--gui", action='store_true')

    parser.add_argument("--render_start_frame", type=int, default=0, help='start frame')

    parser.add_argument("--render_360", type=int, default=-1, help='total num of frames to render')

    parser.add_argument("--K_path", type=str, default='', help='certain path to render')
    parser.add_argument("--cam_path", type=str, default='', help='certain path to render')

    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    return parser
'''
CUDA_HOME=/usr/local/cuda-11.3 CUDA_VISIBLE_DEVICES=0 python mmap_client.py --config logs/NHR/xzq_wmyparams_resv2_u_1e-2/config.py --mmap_path /data/new_disk2/wangla/tmp/libdatachannel_hqh/build/examples/client/mmap --model_path /data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-2/dynamic+99 --render_360 58
'''

parser = config_parser()
args = parser.parse_args()
cfg = mmcv.Config.fromfile(args.config)

n_channel=cfg.fine_model_and_render.rgbnet_dim+1
voxel_size=cfg.voxel_size


# init enviroment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
seed_everything(args)

file_path=args.model_path
os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)

frame_id = args.render_start_frame
data_dict = load_everything_frame(args=args, cfg=cfg, frame_id=frame_id, only_current=True) #可以化简加速

model = dvgo_video.DirectVoxGO_Video()
model.current_frame_id = frame_id
last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgb_net.tar')
ckpt = torch.load(last_ckpt_path)
model.load_rgb_net_mmap(cfg,ckpt)

jsonfile = os.path.join(file_path, f'model_kwargs.json')
with open(jsonfile) as f:
    model_kwargs = json.load(f)

model_kwargs['rgbnet'] = model.rgbnet
dvgo_model=dvgo.DirectVoxGO(**model_kwargs)

def decode():
    frame_count = 0
    while(True):
        jsonfile=os.path.join(file_path,f'newheader_{frame_count}.json')
        with open(jsonfile) as f:
            header = json.load(f)

        with open(os.path.join(file_path, f'mask_{frame_count}.ncrf'), 'rb') as masked_file:
            mask_bits = bitarray()
            mask_bits.fromfile(masked_file) #如果传输的画 直接参考 packet=bitarray(buffer=mm[xx:xxx])
            mask=torch.from_numpy(np.unpackbits(mask_bits).reshape(header['mask_size']).astype(np.bool)).cuda()

        print(f"decode frame {frame_count}")
        with Timer("Decode", device):
            residual_rec_dct = decode_jpeg_huffman(os.path.join(file_path, f'2new_{frame_count}.ncrf'), header, device=device)

        if frame_count==0:
            former_rec=torch.zeros(( mask.size(0), n_channel, voxel_size**3),device=device) #big_data_0
            former_rec[:, 0, :] = former_rec[:, 0, :] - 4.1
            former_rec = recover_misc(residual_rec_dct, former_rec, header, mask,n_channel=n_channel,device=device)
        else:
            deform = np.load(os.path.join(file_path, f'deform_{frame_count}.npy'))
            deform=torch.from_numpy(deform).to(device)
            former_rec = recover_misc_deform(residual_rec_dct, former_rec, header, mask, deform,model_kwargs,
                                                n_channel=n_channel, device=device)

        yield former_rec
        frame_count += 1

mmap_iter=decode()
model_receive=next(mmap_iter)
dvgo_model.density=torch.nn.Parameter(model_receive[:,:1])
dvgo_model.k0.k0=torch.nn.Parameter(model_receive[:,1:])
dvgo_model.k0.eval()

stepsize = cfg.fine_model_and_render.stepsize
render_viewpoints_kwargs = {
    'model': dvgo_model,
    'ndc': cfg.data.ndc,
    'render_kwargs': {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
        'frame_ids': frame_id
    },
}

if args.render_360 > 0:
    if args.K_path=="" or args.cam_path=="":
        render_poses = data_dict['poses'][data_dict['i_train']]
        render_poses = torch.tensor(render_poses).cpu()

        bbox_path = os.path.join(cfg.basedir, cfg.expname, 'bbox.json')
        with open(bbox_path, 'r') as f:
            bbox_json = json.load(f)
        xyz_min_fine = torch.tensor(bbox_json['xyz_min'])
        xyz_max_fine = torch.tensor(bbox_json['xyz_max'])
        bbox = torch.stack([xyz_min_fine, xyz_max_fine]).cpu()

        center = torch.mean(bbox.float(), dim=0)
        up = -torch.mean(render_poses[:, 0:3, 1], dim=0)
        up = up / torch.norm(up)

        radius = torch.norm(render_poses[0, 0:3, 3] - center) * 2
        center = center + up * radius * 0.002

        v = torch.tensor([0, 0, -1], dtype=torch.float32).cpu()
        v = v - up.dot(v) * up
        v = v / torch.norm(v)

        #
        s_pos = center - v * radius - up * radius * 0

        center = center.numpy()
        up = up.numpy()
        radius = radius.item()
        s_pos = s_pos.numpy()

        lookat = center - s_pos
        lookat = lookat / np.linalg.norm(lookat)

        xaxis = np.cross(lookat, up)
        xaxis = xaxis / np.linalg.norm(xaxis)


        sTs = []
        sKs = []
        HWs = []
        frame_ids=[]

        for i in range(0, args.render_360, 1):
            angle = 3.1415926 * 2 * i / 360.0
            pos = s_pos - center
            pos = rodrigues_rotation_matrix(up, -angle).dot(pos)
            pos = pos + center

            lookat = center - pos
            lookat = lookat / np.linalg.norm(lookat)

            xaxis = np.cross(lookat, up)
            xaxis = xaxis / np.linalg.norm(xaxis)

            yaxis = -np.cross(xaxis, lookat)
            yaxis = yaxis / np.linalg.norm(yaxis)

            nR = np.array([xaxis, yaxis, lookat, pos]).T
            nR = np.concatenate([nR, np.array([[0, 0, 0, 1]])])

            sTs.append(nR)
            sKs.append(data_dict['Ks'][data_dict['i_train']][0])
            HWs.append(data_dict['HW'][data_dict['i_train']][0])
            frame_ids.append(i%cfg.frame_num)

        sTs = np.stack(sTs)
        sKs = np.stack(sKs)
    else:
        campose = np.loadtxt("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/CamPose_spiral.inf")
        sTs = campose_to_extrinsic(campose)
        sKs = read_intrinsics("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/Intrinsic_spiral.inf")

        camposes = np.loadtxt("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/CamPose.inf")
        Ts = campose_to_extrinsic(camposes)
        m = np.mean(Ts[:, :3, 3], axis=0)
        print('OBJ center:', m)
        sTs[:, :3, 3] = sTs[:, :3, 3] - m
        sTs[:, :3, 3] = sTs[:, :3, 3] * 2.0 / max(Ts[:, :3, 3].max(), -Ts[:, :3, 3].min())


        frame_ids=[i%cfg.frame_num for i in range(sKs.shape[0])]
        HWs = [(800,800) for _ in range(sKs.shape[0])]

    def model_callback(model, render_kwargs, frame_id):

        if frame_id != args.render_start_frame:
            model_receive = next(mmap_iter)
            model.density = torch.nn.Parameter(model_receive[:, :1])
            model.k0.k0 = torch.nn.Parameter(model_receive[:, 1:])


        return model, render_kwargs


    render_viewpoints_kwargs['model_callback'] = model_callback

    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_360_dyanframes_decode_{args.render_360}')
    os.makedirs(testsavedir, exist_ok=True)
    rgbs, depths = render_viewpoints_frames(
        cfg=cfg,
        render_poses=torch.tensor(sTs).float(),
        HW=HWs,
        Ks=torch.tensor(sKs).float(),frame_ids=frame_ids,
        gt_imgs=None,
        savedir=testsavedir,
        eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
        **render_viewpoints_kwargs)

    imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=10)
    imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30,
                     quality=10)
