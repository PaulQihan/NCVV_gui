from logging import exception
import os, sys, copy, glob, json, time, random, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

#os.environ['CUDA_VISIBLE_DEVICES']='0'
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from shutil import copyfile
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


def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#"/data/new_disk2/wangla/Dataset/NeuralHuman/spiderman/Intrinsic_45000.inf"
#"/data/new_disk2/wangla/Dataset/NeuralHuman/spiderman/CamPose_45000.inf"
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_360", type=int, default=-1, help='total num of frames to render')
    parser.add_argument("--render_start_frame", type=int, default=0, help='start frame')
    # testing options
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", type=int, default=-1)

    parser.add_argument("--K_path", type=str, default='',help='certain path to render')
    parser.add_argument("--cam_path", type=str, default='',help='certain path to render')

    # parser.add_argument("--start_frame", type=int, default=0)
    # parser.add_argument("--end_frame", type=int, default=-1)

    parser.add_argument("--finetune", type=int, default=-1)
    parser.add_argument("--sample_voxels", type=str, default='')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_dyna", action='store_true')
    parser.add_argument("--render_finetune", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--ckpt_name", type=str, default='', help='choose which ckpt, suffix')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser

@torch.no_grad()
def render_viewpoints_frames(model, cfg,render_poses, HW, Ks, frame_ids,ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, model_callback=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    if model_callback is None:
        model_callback = lambda x, y, z: (x, y)

    # if args.rec_ckpt:
    #     savedir += '_rec'
    #     os.makedirs(savedir, exist_ok=True)

    for i, c2w in enumerate(tqdm(render_poses)):

        model, render_kwargs = model_callback(model, render_kwargs, frame_ids[i])

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'rgb_marched_raw']
        rays_o = rays_o.flatten(0, -2).cuda()
        rays_d = rays_d.flatten(0, -2).cuda()
        viewdirs = viewdirs.flatten(0, -2).cuda()

        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i == 0:
            print('Testing', rgb.shape)

        if savedir is not None:
            print(f'Writing images to {savedir}')

            rgb8 = utils.to8b(rgb)
            filename = os.path.join(savedir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)
            depth8 = utils.to8b(1 - depth / np.max(depth))
            filename = os.path.join(savedir, '{:03d}_depth.jpg'.format(i))
            imageio.imwrite(filename, depth8)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    rgbs = np.array(rgbs)
    depths = np.array(depths)

    return rgbs, depths

if __name__ == '__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    cfg.use_res = False
    cfg.use_deform = ""
    cfg.deform_res_mode = ""
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything(args)

    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)

    frame_id = args.render_start_frame
    data_dict = load_everything_frame(args=args, cfg=cfg, frame_id=frame_id, only_current=True)

    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        if cfg.pca_train.use_pca:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_%d_pca.tar' % frame_id)
        elif args.ckpt_name:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname,  args.ckpt_name % frame_id)
            print("loading ", os.path.join(cfg.basedir, cfg.expname,  args.ckpt_name % frame_id))
            print()
            print()
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_%d.tar' % frame_id)
    ckpt_name = ckpt_path.split('/')[-1][:-4]

    model = dvgo_video.DirectVoxGO_Video()
    model.current_frame_id = frame_id
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_rgb_net(cfg)

    # with open(os.path.join(cfg.basedir, cfg.expname,'dynamic+99', f'model_kwargs.json'), 'w') as header_f:
    #     ckpt['model_kwargs']['xyz_min'] = ckpt['model_kwargs']['xyz_min'].tolist()
    #     ckpt['model_kwargs']['xyz_max'] = ckpt['model_kwargs']['xyz_max'].tolist()
    #     ckpt['model_kwargs']['voxel_size_ratio'] = ckpt['model_kwargs']['voxel_size_ratio'].tolist()
    #     ckpt['model_kwargs']['use_res'] = False
    #     json.dump(ckpt['model_kwargs'], header_f, indent=4)

    ckpt['model_kwargs']['rgbnet'] = model.rgbnet
    ckpt['model_kwargs']['cfg'] = cfg
    ckpt['model_kwargs']['use_res'] = cfg.use_res
    ckpt['model_kwargs']['use_deform'] = cfg.use_deform
    ckpt['model_kwargs']['rgbfeat_sigmoid'] = cfg.codec.rgbfeat_sigmoid

    if args.ckpt_name != '':
        ckpt['model_kwargs']['rgbfeat_sigmoid'] = False
    sub_model = dvgo.DirectVoxGO(**ckpt['model_kwargs'])
    if cfg.use_res:
        ckpt['model_state_dict']['k0.former_k0']=sub_model.k0.former_k0 #only for the start frame !!!!!!!!!
    sub_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # sub_model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
    model.dvgos[str(frame_id)] = sub_model.to(device)

    model.dvgos[str(frame_id)].k0.eval()

    # model.activate_refinenet()


    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
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
                # update checkpoint
                if args.ft_path:
                    ckpt_path = args.ft_path
                else:
                    if cfg.pca_train.use_pca:
                        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_%d_pca.tar' % frame_id)
                    elif cfg.use_deform and not args.ckpt_name:
                        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_%d_deform.tar' % frame_id)
                    elif args.ckpt_name:
                        ckpt_path = os.path.join(cfg.basedir, cfg.expname,  args.ckpt_name % frame_id)
                        print("loading ",
                              os.path.join(cfg.basedir, cfg.expname,  args.ckpt_name % frame_id))
                        print()
                        print()
                    else:
                        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last_%d.tar' % frame_id)
                ckpt_name = ckpt_path.split('/')[-1][:-4]

                render_kwargs['frame_ids'] = frame_id
                del model

                print('load',ckpt_path)

                model = dvgo_video.DirectVoxGO_Video()

                ckpt = torch.load(ckpt_path,map_location=device)
                model.current_frame_id = frame_id
                model.load_rgb_net(cfg, exception = True)
                ckpt['model_kwargs']['rgbnet'] = model.rgbnet
                ckpt['model_kwargs']['use_res'] = cfg.use_res
                if cfg.deform_res_mode=="separate":
                    ckpt['model_kwargs']['use_res'] =  cfg.use_res=True
                sub_model = dvgo.DirectVoxGO(**ckpt['model_kwargs'])
                ckpt['model_state_dict'].pop('k0.former_k0',None)
                sub_model.load_state_dict(ckpt['model_state_dict'])
                if cfg.use_res:
                    sub_model.k0.former_k0_cur=sub_model.k0.former_k0

                #ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'svd_rec_density_%d.tar' % frame_id)
                #sub_model.density.data = torch.load(ckpt_path).unsqueeze(0)

                #ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'svd_rec_k0_%d.tar' % frame_id)
                #tmp = torch.load(ckpt_path).unsqueeze(0)
                #sub_model.k0.data[:,:3,:,:,:] = tmp[:,:3,:,:,:]

                model.dvgos[str(frame_id)] = sub_model.to(device)

            #model.activate_refinenet()

            return model, render_kwargs


        render_viewpoints_kwargs['model_callback'] = model_callback

        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_360_dyanframes_{ckpt_name}_{args.render_360}')
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

