
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
from render_dyna import render_viewpoint_frame, rodrigues_rotation_matrix,read_intrinsics,campose_to_extrinsic

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='./logs/NHR/xzq_wmyparams_deform_tv_res_cube_l1/config.py',
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')

    parser.add_argument("--mmap_path", type=str, default='',
                        help='mmap file path')

    parser.add_argument("--model_path", type=str, default='./logs/NHR/xzq_wmyparams_deform_tv_res_cube_l1/dynamic+99',
                        help='mask file path')

    parser.add_argument("--render_start_frame", type=int, default=0, help='start frame')

    parser.add_argument("--render_360", type=int, default=92, help='total num of frames to render')

    parser.add_argument("--K_path", type=str, default='', help='certain path to render')
    parser.add_argument("--cam_path", type=str, default='', help='certain path to render')

    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')


    #---- GUI config
    parser.add_argument("--gui", action='store_true')
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument('--W', type=int, default=800)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--fovy', type=int, default=60)
    parser.add_argument('--test', default=True)

    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--dt_gamma', type=float, default=1/128)
    parser.add_argument('--max_steps', type=int, default=1024)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    return parser
'''
CUDA_HOME=/usr/local/cuda-11.3 CUDA_VISIBLE_DEVICES=0 python mmap_client.py --config logs/NHR/xzq_wmyparams_resv2_u_1e-2/config.py --mmap_path /data/new_disk2/wangla/tmp/libdatachannel_hqh/build/examples/client/mmap --model_path /data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_wmyparams_resv2_u_1e-2/dynamic+99 --render_360 58
'''




    

class ncvv:
    def __init__(self):
        self.parser = config_parser()
        self.args = self.parser.parse_args()
        self.cfg = mmcv.Config.fromfile(self.args.config)

        self.n_channel=self.cfg.fine_model_and_render.rgbnet_dim+1
        self.voxel_size=self.cfg.voxel_size

        # init enviroment
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        seed_everything(self.args)

        self.file_path=self.args.model_path
        os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname), exist_ok=True)

        self.frame_id = self.args.render_start_frame
        self.data_dict = load_everything_frame(args=self.args, cfg=self.cfg, frame_id=self.frame_id, only_current=True) #可以化简加速

        self.model = dvgo_video.DirectVoxGO_Video()
        self.model.current_frame_id = self.frame_id
        self.last_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, f'rgb_net.tar')
        self.ckpt = torch.load(self.last_ckpt_path)
        self.model.load_rgb_net_mmap(self.cfg, self.ckpt)

        self.jsonfile = os.path.join(self.file_path, f'model_kwargs.json')
        with open(self.jsonfile) as f:
            self.model_kwargs = json.load(f)

        self.model_kwargs['rgbnet'] = self.model.rgbnet
        self.dvgo_model=dvgo.DirectVoxGO(**self.model_kwargs)

        self.mmap_iter=self.decode()
        self.model_receive=next(self.mmap_iter)
        self.dvgo_model.density=torch.nn.Parameter(self.model_receive[:,:1])
        self.dvgo_model.k0.k0=torch.nn.Parameter(self.model_receive[:,1:])
        self.dvgo_model.k0.eval()

        self.stepsize = self.cfg.fine_model_and_render.stepsize
        self.render_viewpoints_kwargs = {
            'model': self.dvgo_model,
            'ndc': self.cfg.data.ndc,
            'render_kwargs': {
                'near': self.data_dict['near'],
                'far': self.data_dict['far'],
                'bg': 1 if self.cfg.data.white_bkgd else 0,
                'stepsize': self.stepsize,
                'inverse_y': self.cfg.data.inverse_y,
                'flip_x': self.cfg.data.flip_x,
                'flip_y': self.cfg.data.flip_y,
                'render_depth': True,
                'frame_ids': self.frame_id
            },
        }
        self.render_poses = self.data_dict['poses'][self.data_dict['i_train']]
        self.render_poses = torch.tensor(self.render_poses).cpu()

        self.bbox_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'bbox.json')
        with open(self.bbox_path, 'r') as f:
            self.bbox_json = json.load(f)
        self.xyz_min_fine = torch.tensor(self.bbox_json['xyz_min'])
        self.xyz_max_fine = torch.tensor(self.bbox_json['xyz_max'])
        self.bbox = torch.stack([self.xyz_min_fine, self.xyz_max_fine]).cpu()

        self.center = torch.mean(self.bbox.float(), dim=0)
        self.up = -torch.mean(self.render_poses[:, 0:3, 1], dim=0)
        self.up = self.up / torch.norm(self.up)

        self.radius = torch.norm(self.render_poses[0, 0:3, 3] - self.center) * 2
        self.center = self.center + self.up * self.radius * 0.002

        self.v = torch.tensor([0, 0, -1], dtype=torch.float32).cpu()
        self.v = self.v - self.up.dot(self.v) * self.up
        self.v = self.v / torch.norm(self.v)

        #
        self.s_pos = self.center - self.v * self.radius - self.up * self.radius * 0

        self.center = self.center.numpy()
        self.up = self.up.numpy()
        self.radius = self.radius.item()
        self.s_pos = self.s_pos.numpy()

        self.lookat = self.center - self.s_pos
        self.lookat = self.lookat / np.linalg.norm(self.lookat)

        self.xaxis = np.cross(self.lookat, self.up)
        self.xaxis = self.xaxis / np.linalg.norm(self.xaxis)
        self.angle = 0
        self.pos = self.s_pos - self.center
        self.pos = rodrigues_rotation_matrix(self.up, -self.angle).dot(self.pos)
        self.pos = self.pos + self.center

        self.lookat = self.center - self.pos
        self.lookat = self.lookat / np.linalg.norm(self.lookat)

        self.xaxis = np.cross(self.lookat, self.up)
        self.xaxis = self.xaxis / np.linalg.norm(self.xaxis)

        self.yaxis = -np.cross(self.xaxis, self.lookat)
        self.yaxis = self.yaxis / np.linalg.norm(self.yaxis)

        self.nR = np.array([self.xaxis, self.yaxis, self.lookat, self.pos]).T
        self.nR = np.concatenate([self.nR, np.array([[0, 0, 0, 1]])])

        self.sT = self.nR
        self.sK = self.data_dict['Ks'][self.data_dict['i_train']][0]
        self.HW = self.data_dict['HW'][self.data_dict['i_train']][0]
        self.frame_id = 0
        self.render_viewpoints_kwargs['model_callback'] = self.model_callback



        if self.args.gui:
            from gui_module.gui import NeRFGUI
            render_func = self.gui_render_func
            gui = NeRFGUI(self.args, render_func=render_func)
            gui.render()
        else:
            self.testsavedir = os.path.join(self.cfg.basedir, self.cfg.expname, f'render_360_dyanframes_decode_{self.args.render_360}')
            os.makedirs(self.testsavedir, exist_ok=True)
        
        
    
            # else:
            #     campose = np.loadtxt("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/CamPose_spiral.inf")
            #     sTs = campose_to_extrinsic(campose)
            #     sKs = read_intrinsics("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/Intrinsic_spiral.inf")

            #     camposes = np.loadtxt("/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam/CamPose.inf")
            #     Ts = campose_to_extrinsic(camposes)
            #     m = np.mean(Ts[:, :3, 3], axis=0)
            #     print('OBJ center:', m)
            #     sTs[:, :3, 3] = sTs[:, :3, 3] - m
            #     sTs[:, :3, 3] = sTs[:, :3, 3] * 2.0 / max(Ts[:, :3, 3].max(), -Ts[:, :3, 3].min())


            #     frame_ids=[i%cfg.frame_num for i in range(sKs.shape[0])]
            #     HWs = [(800,800) for _ in range(sKs.shape[0])]



    def decode(self):
        self.frame_count = 0
        while(True):
            self.jsonfile=os.path.join(self.file_path,f'newheader_{self.frame_count}.json')
            with open(self.jsonfile) as f:
                self.header = json.load(f)

            with open(os.path.join(self.file_path, f'mask_{self.frame_count}.ncrf'), 'rb') as masked_file:
                self.mask_bits = bitarray()
                self.mask_bits.fromfile(masked_file) #如果传输的画 直接参考 packet=bitarray(buffer=mm[xx:xxx])
                self.mask=torch.from_numpy(np.unpackbits(self.mask_bits).reshape(self.header['mask_size']).astype(np.bool)).cuda()

            print(f"decode frame {self.frame_count}")
            with Timer("Decode", self.device):
                self.residual_rec_dct = decode_jpeg_huffman(os.path.join(self.file_path, f'2new_{self.frame_count}.ncrf'), self.header, device=self.device)

            if self.frame_count==0:
                self.former_rec=torch.zeros(( self.mask.size(0), self.n_channel, self.voxel_size**3),device=self.device) #big_data_0
                self.former_rec[:, 0, :] = self.former_rec[:, 0, :] - 4.1
                self.former_rec = recover_misc(self.residual_rec_dct, self.former_rec, self.header, self.mask,n_channel=self.n_channel,device=self.device)
            else:
                self.deform = np.load(os.path.join(self.file_path, f'deform_{self.frame_count}.npy'))
                self.deform=torch.from_numpy(self.deform).to(self.device)
                self.former_rec = recover_misc_deform(self.residual_rec_dct, self.former_rec, self.header, self.mask, self.deform,self.model_kwargs,
                                                    n_channel=self.n_channel, device=self.device)

            yield self.former_rec
            self.frame_count += 1       
    
    def update_pose(self, delta_x, delta_y, t):
        self.angle_x = 3.1415926 * 2 * delta_x / 360.0
        self.angle_y = 3.1415926 * 2 * delta_y / 360.0
        self.pos = self.s_pos - self.center
        self.pos = rodrigues_rotation_matrix(self.up, -self.angle_x).dot(self.pos)
        self.pos = rodrigues_rotation_matrix(self.xaxis, -self.angle_y).dot(self.pos)
        self.pos = self.pos + self.center

        self.lookat = self.center - self.pos
        self.lookat = self.lookat / np.linalg.norm(self.lookat)

        self.xaxis = np.cross(self.lookat, self.up)
        self.xaxis = self.xaxis / np.linalg.norm(self.xaxis)

        self.yaxis = -np.cross(self.xaxis, self.lookat)
        self.yaxis = self.yaxis / np.linalg.norm(self.yaxis)

        self.nR = np.array([self.xaxis, self.yaxis, self.lookat, self.pos]).T
        self.nR = np.concatenate([self.nR, np.array([[0, 0, 0, 1]])])

        self.sT = self.nR
        self.sK = self.data_dict['Ks'][self.data_dict['i_train']][0]
        self.HW = self.data_dict['HW'][self.data_dict['i_train']][0]
        self.frame_id = t%self.cfg.frame_num

    def model_callback(self, model, render_kwargs, frame_id):

            if self.frame_id != self.args.render_start_frame:
                self.model_receive = next(self.mmap_iter)
                self.model.density = torch.nn.Parameter(self.model_receive[:, :1])
                self.model.k0 = torch.nn.Parameter(self.model_receive[:, 1:])


            return model, render_kwargs

    def render(self):
        self.rgb, self.depth = render_viewpoint_frame(
                cfg=self.cfg,
                render_pose=torch.tensor(self.sT).float(), 
                HW=self.HW,
                K=torch.tensor(self.sK).float(),
                frame_id=self.frame_id,
                gt_imgs=None,
                savedir=self.testsavedir,
                eval_ssim=self.args.eval_ssim, eval_lpips_alex=self.args.eval_lpips_alex, eval_lpips_vgg=self.args.eval_lpips_vgg,
                **self.render_viewpoints_kwargs)
    
    @torch.no_grad()
    def gui_render_func(self,
                    render_pose,
                    inp_K,
                    W, H,
                    time,
                    bg_color = None,
                    spp = None,
                    downscale = None,
                    camera_k =None,
                    **gui_kwargs):
        
        HW = [H,W]

        K = torch.eye(4)
        K[0,0], K[1,1] = inp_K[0], inp_K[1]
        K[0,2], K[1,2] = inp_K[2], inp_K[3]

        # print('--- K : ', K)
        # print('--- self.sK : ', self.sK)

        # exit(0)

        # render_pose = self.sT
        # K = self.sK

        frame_id = int(time * self.cfg.frame_num)
        # self.render_viewpoints_kwargs['frame_ids'] = frame_id

        # print('---- self.frame_id : ',self.frame_id)
        # exit(0)
        
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            rgb, depth = render_viewpoint_frame(
                            cfg=self.cfg,
                            render_pose=torch.tensor(render_pose).float(), 
                            HW=HW,
                            K=K.float(),
                            frame_id=frame_id,
                            mute=True,
                            **self.render_viewpoints_kwargs)
        outputs = {'image': rgb, 'depth': depth}
        return outputs

        # self.render_viewpoints_kwargs = {
        #     'model': self.dvgo_model,
        #     'ndc': self.cfg.data.ndc,
        #     'render_kwargs': {
        #         'near': self.data_dict['near'],
        #         'far': self.data_dict['far'],
        #         'bg': 1 if self.cfg.data.white_bkgd else 0,
        #         'stepsize': self.stepsize,
        #         'inverse_y': self.cfg.data.inverse_y,
        #         'flip_x': self.cfg.data.flip_x,
        #         'flip_y': self.cfg.data.flip_y,
        #         'render_depth': True,
        #         'frame_ids': self.frame_id
        #     },



        # self.cam.pose, self.cam.intrinsics, self.W, self.H, self.time, self.bg_color, self.spp, self.downscale)
            
if __name__ == "__main__":
    model = ncvv()
    # model.demo.show()
    print("done")

    'running example see : cmd_line.sh'