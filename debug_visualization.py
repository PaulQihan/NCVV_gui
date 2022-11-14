from logging import exception
import os, sys, copy, glob, json, time, random, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

os.environ['CUDA_HOME']='/usr/local/cuda-11.3'
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import gc
import ipdb

from tools.voxelized import sample_grid_on_voxel
from lib import utils, dvgo, dmpigo, dvgo_video
from lib.load_data import load_data, load_data_frame
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
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
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--render_360", type=int, default=-1)
    parser.add_argument("--render_360_step", type=int, default=1)
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", type=int, default=-1)

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)

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
    parser.add_argument("--ckpt_name", type=str, default='', help='choose which ckpt')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, model_callback = None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
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
        model_callback = lambda x,y,z:(x,y)

    # if args.rec_ckpt:
    #     savedir += '_rec'
    #     os.makedirs(savedir, exist_ok=True)

    for i, c2w in enumerate(tqdm(render_poses)):
        model,render_kwargs = model_callback(model,render_kwargs,i)


        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth','rgb_marched_raw']
        rays_o = rays_o.flatten(0,-2).cuda()
        rays_d = rays_d.flatten(0,-2).cuda()
        viewdirs = viewdirs.flatten(0,-2).cuda()

    
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

        if savedir is not None:

            print(f'Writing images to {savedir}')
            
            rgb8 = utils.to8b(rgb)
            filename = os.path.join(savedir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)
            depth8 =  utils.to8b(1 - depth / np.max(depth))
            filename = os.path.join(savedir, '{:03d}_depth.jpg'.format(i))
            imageio.imwrite(filename, depth8)

        if gt_imgs is not None and render_factor==0:
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


def seed_everything(args):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'frame_ids'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def load_everything_frame(args, cfg, frame_id, only_current = False,scale = 1.0):
    '''Load images / poses / camera settings / data split.
    '''

    data_dict = load_data_frame(cfg.data, frame_id, only_current = only_current,scale=scale)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test','i_replay','i_current', 'irregular_shape',
            'poses', 'render_poses', 'images', 'frame_ids'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(model, args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage,
                             coarse_ckpt_path=None, fix_rgb = False, use_pca = False ,deform_res_stage=''):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        print("xyz shift type",xyz_shift)
        print("xyz min type",xyz_min)
        xyz_min =xyz_min.float()- xyz_shift
        xyz_max =xyz_max.float()+ xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test,i_replay, i_current, poses, render_poses, images, frame_ids = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'i_replay','i_current','poses', 'render_poses', 'images', 'frame_ids'
        ]
    ]
    #ipdb.set_trace()
    frame_ids = frame_ids.cpu()
    unique_frame_ids = torch.unique(frame_ids, sorted=True).cpu().numpy().tolist()

    current_frame= frame_ids[-1].item()
    #current_ray_mask = (frame_ids==current_frame)

    #if stage=='coarse':
    #    i_train = i_train[i_current]
        #HW = HW[i_current]
        #Ks = Ks[i_current]

    N_iters =cfg_train.N_iters if current_frame==0 else cfg_train.N_iters_pretrained

    if deform_res_stage:
        print("-------deform_res_stage------------", deform_res_stage)
    print('**** Frame id: %d **********' % current_frame,stage)

    use_deform = cfg.use_deform if (current_frame != 0 and stage == "fine") else ""


    if use_pca:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d_pca.tar' % current_frame)
    elif use_deform or deform_res_stage=="deform":
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d_deform.tar' % current_frame)
    elif args.ckpt_name != '':
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, args.ckpt_name % current_frame)
        print("loading ", os.path.join(cfg.basedir, cfg.expname, args.ckpt_name % current_frame))
        print()
        print()
    else:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d.tar' % current_frame)
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None


    if use_deform=="grid" or deform_res_stage=="deform":
        # cfg_train["lrate_density"] = 0
        # cfg_train["lrate_k0"] = 0
        # cfg_train["lrate_rgbnet"] = 0

        use_deform_tmp = use_deform + deform_res_stage
    else:
        use_deform_tmp=''

    if cfg.use_res or deform_res_stage == "res":
        #cfg_train["lrate_deformation_field"] = 0
        use_res_tmp = str(cfg.use_res) + deform_res_stage
    else:
        use_res_tmp = ''

    if reload_ckpt_path is None:
        start = 0

        sub_model  = model.create_current_model(current_frame, xyz_min, xyz_max,stage, cfg, cfg_model, cfg_train, coarse_ckpt_path,
                                                use_pca = use_pca,use_res=use_res_tmp,use_deform=use_deform_tmp)
        if cfg_model.maskout_near_cam_vox: #true
            sub_model.maskout_near_cam_vox(poses[i_current,:3,3], near)

        if deform_res_stage=="deform":

            pretrain_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d.tar' % (current_frame - 1))
            print('load pretrained model ', pretrain_ckpt_path)
            if current_frame == 1:
                sub_model.load_pretrain_deform(pretrain_ckpt_path, current_frame,True)
            else:
                sub_model.load_pretrain_deform_res(pretrain_ckpt_path, deform_res_stage)
            start = 0
        elif deform_res_stage=="res":
            pretrain_ckpt_path = os.path.join(cfg.basedir, cfg.expname,
                                              f'{stage}_last_%d_deform.tar' % (current_frame))
            print('load pretrained model ', pretrain_ckpt_path)

            sub_model.load_pretrain_deform_res(pretrain_ckpt_path, deform_res_stage,cfg.deform_low_reso)
            start = 0
        elif cfg.use_res and current_frame > 0 and stage == 'fine':
            pretrain_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d.tar' % (current_frame - 1))
            print('load pretrained model ', pretrain_ckpt_path)
            sub_model.load_pretrain(pretrain_ckpt_path,current_frame)
            start = 0
        elif use_deform=="grid":
            if current_frame == 1 or cfg.deform_from_start:
                pretrain_ckpt_path = os.path.join(cfg.basedir, cfg.expname,
                                                  f'{stage}_last_%d.tar' % (0))
                print('load pretrained model ', pretrain_ckpt_path)

            else:
                pretrain_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last_%d_deform.tar' % (current_frame - 1))
                print('load pretrained model ', pretrain_ckpt_path)

            sub_model.load_pretrain_deform(pretrain_ckpt_path, current_frame,cfg.deform_from_start)
            start = 0

    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        else:
            model_class = dvgo.DirectVoxGO
        #model = utils.load_model(model_class, reload_ckpt_path).to(device)
        ckpt = torch.load(reload_ckpt_path)
        start = ckpt['global_step']
        if start >= N_iters:
            return
        model_kwargs= ckpt['model_kwargs']
        model_kwargs['rgbnet'] = model.rgbnet
        model_kwargs['cfg'] = cfg
        #model_kwargs['fine'] = (stage=='fine')
        sub_model = model_class(**model_kwargs)
        sub_model.load_state_dict(ckpt['model_state_dict'])
        model.current_frame_id = current_frame
        model.dvgos[str(model.current_frame_id)] = sub_model

    if start >= N_iters:
        return

        
    if stage=='coarse':
        model.set_dvgo_update([current_frame])
    else:
        model.set_dvgo_update(unique_frame_ids)


    optimizer = utils.create_optimizer_or_freeze_model_frame(model, cfg_train, global_step=0, fix_rgb= fix_rgb,
                                                             deform_stage=use_deform_tmp,res_stage=use_res_tmp)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }


    log_file_path=os.path.join(cfg.basedir, cfg.expname, f'log_{stage}_%d.txt' % (current_frame))
    log_ptr = open(log_file_path, "a+")


    # init batch rays sampler
    def gather_training_rays():

        rgb_tr_s = []
        rays_o_tr_s = []
        rays_d_tr_s = []
        viewdirs_tr_s = []
        imsz_s = []
        frame_id_tr = []
        for id in unique_frame_ids:
            if stage=='coarse':
                if id != current_frame:
                    continue
            #id_mask = (frame_ids==id)
            t_train = i_train
            if data_dict['irregular_shape']:
                rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train ]
            else:
                rgb_tr_ori = images[t_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            if cfg_train.ray_sampler == 'in_maskcache':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr = dvgo.get_training_rays_in_maskcache_sampling(
                        rgb_tr_ori=rgb_tr_ori,
                        train_poses=poses[t_train],
                        HW=HW[t_train], Ks=Ks[t_train],
                        ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                        model=model.dvgos[str(id)], frame_ids = frame_ids[t_train], render_kwargs=render_kwargs)
            elif cfg_train.ray_sampler == 'flatten':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz,frame_ids_tr = dvgo.get_training_rays_flatten(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[t_train],
                    HW=HW[t_train], Ks=Ks[t_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    frame_ids = frame_ids[t_train])
            else:
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_ids_tr = dvgo.get_training_rays(
                    rgb_tr=rgb_tr_ori,
                    train_poses=poses[t_train],
                    HW=HW[t_train], Ks=Ks[t_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    frame_ids = frame_ids[t_train])
            rgb_tr_s.append(rgb_tr)
            rays_o_tr_s.append(rays_o_tr)
            rays_d_tr_s.append(rays_d_tr)
            viewdirs_tr_s.append(viewdirs_tr)
            imsz_s.append(imsz)
            frame_id_tr.append(frame_ids_tr)

        rgb_tr_s = torch.cat(rgb_tr_s)
        rays_o_tr_s = torch.cat(rays_o_tr_s)
        rays_d_tr_s = torch.cat(rays_d_tr_s)
        viewdirs_tr_s = torch.cat(viewdirs_tr_s)
        imsz_tmp = []
        for imsz in imsz_s:
            imsz_tmp = imsz_tmp + imsz
        imsz_s = imsz_tmp
        frame_id_tr = torch.cat(frame_id_tr)



        index_generator = dvgo.batch_indices_generator(len(rgb_tr_s), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr_s, rays_o_tr_s, rays_d_tr_s, viewdirs_tr_s, imsz_s, frame_id_tr, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler = gather_training_rays()
    frame_id_tr = frame_id_tr.cpu()
    # view-count-based learning rate


    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    psnr_raw = []
    loss_subs = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+N_iters):
        # renew occupancy grid
        if sub_model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            if use_res_tmp:
                self_alpha = F.max_pool3d(sub_model.activate_density(sub_model.density+sub_model.former_density_cur), kernel_size=3, padding=1, stride=1)[0,0]
            else:
                self_alpha = F.max_pool3d(sub_model.activate_density(sub_model.density), kernel_size=3, padding=1, stride=1)[0,0]
            sub_model.mask_cache.mask &= (self_alpha > sub_model.fast_color_thres)

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale and current_frame==0:
            #if current_frame==0 or stage =='coarse':
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(sub_model, dvgo.DirectVoxGO):
                sub_model.scale_volume_grid(cur_voxels)
            elif isinstance(sub_model, dmpigo.DirectMPIGO):
                sub_model.scale_volume_grid(cur_voxels, sub_model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model_frame(model, cfg_train, global_step=global_step, fix_rgb= fix_rgb,
                                                             deform_stage=use_deform_tmp,res_stage=use_res_tmp)
            if not use_deform_tmp:
                sub_model.density.data.sub_(1.3)

        if global_step in cfg_train.pg_scale_pretrained and current_frame>0:
            #if current_frame==0 or stage =='coarse':
            n_rest_scales = len(cfg_train.pg_scale_pretrained)-cfg_train.pg_scale_pretrained.index(global_step)-1
            
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(sub_model, dvgo.DirectVoxGO):
                sub_model.scale_volume_grid(cur_voxels)
            elif isinstance(sub_model, dmpigo.DirectMPIGO):
                sub_model.scale_volume_grid(cur_voxels, sub_model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model_frame(model, cfg_train, global_step=global_step, fix_rgb= fix_rgb,
                                                             deform_stage=use_deform_tmp,res_stage=use_res_tmp)
            if not use_deform_tmp:
                sub_model.density.data.sub_(1.3)


        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            frameids = frame_id_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            frameids = frame_id_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError


        #if (frameids==current_frame).sum()>0:
        #ipdb.set_trace()
        sorted_rays_o = []
        sorted_rays_d = []
        sorted_viewdirs = []
        sorted_frameids = []
        sorted_target = []
        for id in unique_frame_ids:
            #if id != 0:
            #    continue
            mask = frameids==id
            sorted_rays_o.append(rays_o[mask,:])
            sorted_rays_d.append(rays_d[mask,:])
            sorted_viewdirs.append(viewdirs[mask,:])
            sorted_frameids.append(frameids[mask])
            sorted_target.append(target[mask,:])
        rays_o = torch.cat(sorted_rays_o,dim=0)
        rays_d = torch.cat(sorted_rays_d,dim=0)
        viewdirs = torch.cat(sorted_viewdirs,dim=0)
        target = torch.cat(sorted_target,dim=0)
        frameids = torch.cat(sorted_frameids)


        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
       
        # volume rendering
        frameids = frameids.long()

        render_result = model(rays_o, rays_d, viewdirs,frameids, global_step=global_step, **render_kwargs)
        
        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        # print(render_result['rgb_marched'].size()) #[20384, 3]
        # print(target.size()) # [10192, 3]
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)

        psnr = utils.mse2psnr(loss.detach())

        if cfg.res_lambda != 0 and current_frame != 0 and stage=='fine' and (cfg.use_res or deform_res_stage=="res"):
            loss_l1 = cfg.res_lambda * F.l1_loss(sub_model.k0.k0, torch.zeros_like(sub_model.k0.k0,device=sub_model.k0.device))
            #loss_l1 =loss_l1+ cfg.res_density_lambda * F.l1_loss(sub_model.density, torch.zeros_like(sub_model.density,device=sub_model.density.device)) not effective
            loss = loss + loss_l1

        if cfg.deform_lambda != 0 and current_frame != 0 and stage=='fine' and (cfg.use_res or deform_res_stage=="deform"):
            loss_l1 = cfg.deform_lambda * F.l1_loss(sub_model.deformation_field, torch.zeros_like(sub_model.deformation_field,device=sub_model.k0.device))
            loss = loss + loss_l1

        if 'rgb_marched_raw' in render_result:
            loss2 = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched_raw'], target)
            loss = (loss + loss2)
            psnrraw = utils.mse2psnr(loss2.detach())
            psnr_raw.append(psnrraw.item())


        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        loss_sub = { id:[] for id in unique_frame_ids}
        pred = render_result['rgb_marched'].detach()

        for id in unique_frame_ids:
            mask = frameids==id
            if mask.sum()==0:
                continue
            tmp = cfg_train.weight_main * F.mse_loss(pred[mask,:], target[mask,:])
            loss_sub[id].append(utils.mse2psnr(tmp.detach()).cpu())


        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_deform > 0 and use_deform_tmp:
                sub_model.deform_total_variation_add_grad(
                    cfg_train.weight_tv_deform / len(rays_o), global_step < cfg_train.tv_dense_before)
            if cfg_train.weight_tv_density>0 and deform_res_stage!="deform":
                sub_model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0 and deform_res_stage!="deform":
                sub_model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if (global_step-1)%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'

            #ipdb.set_trace()
            
            if len(psnr_raw)>0:
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / PSNR_RAW: {np.mean(psnr_raw):5.2f} '
                       f'Eps: {eps_time_str} ')
                psnr_raw = []

                print(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / PSNR_RAW: {np.mean(psnr_raw):5.2f} '
                       f'Eps: {eps_time_str} ', file=log_ptr)
                log_ptr.flush()

            elif cfg.res_lambda != 0 and current_frame != 0 and stage=='fine' and (cfg.use_res or deform_res_stage=="res" ):
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} /  Loss_L1: {loss_l1.item():.9f} /PSNR: {np.mean(psnr_lst):5.2f} / '
                           f'Eps: {eps_time_str} ')
                print(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} /  Loss_L1: {loss_l1.item():.9f} /PSNR: {np.mean(psnr_lst):5.2f} / '
                           f'Eps: {eps_time_str} ', file=log_ptr)
                log_ptr.flush()

            elif cfg.deform_lambda != 0 and current_frame != 0 and stage=='fine' and (cfg.use_res or deform_res_stage=="deform" ):
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} /  Loss_L1: {loss_l1.item():.9f} /PSNR: {np.mean(psnr_lst):5.2f} / '
                           f'Eps: {eps_time_str} ')
                print(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} /  Loss_L1: {loss_l1.item():.9f} /PSNR: {np.mean(psnr_lst):5.2f} / '
                           f'Eps: {eps_time_str} ', file=log_ptr)
                log_ptr.flush()
            else:
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                        f'Eps: {eps_time_str} ')

                print(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                        f'Eps: {eps_time_str} ', file=log_ptr)
                log_ptr.flush()


            for id in unique_frame_ids:
                print(id, 'psnr:',torch.mean(torch.tensor(loss_sub[id])))

            psnr_lst = []
            loss_sub = { id:[] for id in unique_frame_ids}

        if global_step%args.i_weights==0:
            print('save checkpoint 1')
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if use_res_tmp:
        sub_model.density = torch.nn.Parameter(sub_model.density + sub_model.former_density)
        sub_model.former_density = None
        print("add")

    if global_step != -1:
        if stage=='fine':
            return

        torch.save({
            'global_step': global_step,
            'model_kwargs': sub_model.get_kwargs(),
            'model_state_dict': sub_model.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
        
if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
        # torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    seed_everything(args)

    # load images / poses / camera settings / data split


    # export scene bbox and camera poses in 3d for debugging and visualization

    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')

        args.datadir = '/home/vrlab/workspace/NCVV/NCVV/logs/NHR/jywq_2_250'

        data_dict = load_everything_frame(args=args, cfg=cfg, frame_id=0, only_current=True)

        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()