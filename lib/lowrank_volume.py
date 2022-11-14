import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import ipdb
import time
from  .voxel_utils import Merge_Volume_CUDA


def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)



class TensorBase(torch.nn.Module):
    def __init__(self, gridSize, device, feat_dim = 48*4, cfg = None ):
        super(TensorBase, self).__init__()

        self.cfg = cfg

        self.feat_dim = feat_dim
        self.device=device

        self.update_stepSize(gridSize)
        self.init_svd_volume()



    

    def update_stepSize(self, gridSize):
      
        print("grid size", gridSize)
        self.gridSize= torch.LongTensor(gridSize).to(self.device)

    def init_svd_volume(self):
        pass

    def compute_features(self, xyz_sampled):
        pass
    

    def get_kwargs(self):
        return {
            'gridSize':self.gridSize.tolist(),
            'feat_dim': self.feat_dim,

        }



    def forward(self, ray_pts):
        feature = self.compute_features(ray_pts)
        return feature



class TensorCP(TensorBase):

    def __init__(self,  gridSize, device, **kargs):
        super(TensorCP, self).__init__(gridSize, device, **kargs)


        #self.basis_mat = torch.nn.Linear(self.num_componet*3*self.level, self.feat_dim, bias=False).to(device)
        

    def update_stepSize(self, gridSize):
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.gridSize = self.gridSize *1
        print("grid size", self.gridSize)


    def init_svd_volume(self):
        self.level = 1
        self.base = 1.5
        #self.num_componet = 12
        self.num_componet  = self.feat_dim  # //3 // self.level
        self.real_dim = self.feat_dim # //3 // self.level
        self.line = []
        for l in range(self.level):
            self.line+=(self.init_one_svd(self.num_componet, (self.gridSize/(self.base**l)).long(), 0.01))
        self.line = torch.nn.ParameterList(self.line).to(self.device)


    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        for i in range(3):
            vec_id = i
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return line_coef

    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        
        feat_res = []
        cnt = 0
        for l in range(self.level):
            for i in range(3):
                tmp = F.grid_sample(self.line[cnt], coordinate_line[[i]], align_corners=True).reshape(self.num_componet,-1).T
                feat_res.append(tmp)
                cnt = cnt + 1

            #feat_res.append(torch.cat(tmpl, dim=1))

        feature = feat_res[0]
        for i in range(1, len(feat_res)):
            feature = feature*feat_res[i]

        #feature = self.basis_mat(feature)

        #feature = torch.cat(feat_res, dim=1)
        #feature = self.basis_mat(feature)
        return feature
    
    

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):

        cnt = 0
        for l in range(self.level):
            for i in range(3):
                vec_id = i
                #print(cnt, line[i].size())
                line[cnt] = torch.nn.Parameter(
                    F.interpolate(line[cnt].data, size=(int(res_target[vec_id]/(self.base**l)), 1), mode='bilinear', align_corners=True))
                
                cnt = cnt + 1
        return line

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.line = self.up_sampling_Vector(self.line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')



class TensorVP(TensorBase):

    def __init__(self,  gridSize, device, **kargs):
        super(TensorVP, self).__init__(gridSize, device, **kargs)


        #self.basis_mat = torch.nn.Linear(self.num_componet*3*self.level, self.feat_dim, bias=False).to(device)
        

    def update_stepSize(self, gridSize):
        self.gridSize= int(max(gridSize))*2
        print("TensorVP grid size:", self.gridSize)


    def init_svd_volume(self):
        self.level = 1
        self.base = 1.5
        #self.num_componet = 12
        self.num_basises = 29
        #basises = torch.rand(self.num_basises,3)*2.0-1.0
        basises = torch.tensor(fibonacci_sphere(self.num_basises)).float()
        basises = torch.nn.functional.normalize(basises)*math.sqrt(3.0)
        self.register_buffer('basises', basises)
        self.num_componet  = self.feat_dim  //(self.num_basises+3) // self.level
        #self.num_componet  = 2
        self.real_dim = self.get_feature_size() 
        self.line = []
        for l in range(self.level):
            self.line+=(self.init_one_svd(self.num_componet, int(self.gridSize/(self.base**l)), 0.01))

        for i in range(3):
            self.line.append(
                torch.nn.Parameter(0.0001 * torch.randn((1, self.num_componet, self.gridSize, 1))))
        self.line = torch.nn.ParameterList(self.line).to(self.device)


    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []


        for i in range(self.num_basises):
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize, 1))))

    
        return line_coef

    def projection(self, xyz_sampled, basis_ind):
        shape = xyz_sampled.size()[:-1]
        assert xyz_sampled.size(-1)==3

        xyz = xyz_sampled.reshape(-1,3)

        

        basis = self.basises[basis_ind]

        res = torch.matmul(xyz,basis)
        res = res/ (torch.linalg.vector_norm(basis)**2)
       
        res = res.unsqueeze(-1)

        return res

    def get_feature_size(self):
        return self.num_componet*(self.num_basises+3)

    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):
        feat_res = []
        for i in range(self.num_basises):
            coords = self.projection(xyz_sampled,i)

            

            coordinate_line = coords[..., 0].unsqueeze(0)
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(1, -1, 1, 2)


            
            for l in range(self.level):
                    cnt = l*self.num_basises + i
                   
                    tmp = F.grid_sample(self.line[cnt], coordinate_line[[0]], align_corners=True).reshape(self.num_componet,-1).T
                    feat_res.append(tmp)

        coordinate_line = torch.stack((xyz_sampled[..., 0], xyz_sampled[..., 1], xyz_sampled[..., 2]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        for l in range(3):
            cnt = self.level*self.num_basises + l
            tmp = F.grid_sample(self.line[cnt], coordinate_line[[l]], align_corners=True).reshape(self.num_componet,-1).T
            feat_res.append(tmp)
          

        #feature = feat_res[0]
        #for i in range(1, len(feat_res)):
        #    feature = feature*feat_res[i]

        #feature = self.basis_mat(feature)

        feature = torch.cat(feat_res, dim=1)
        #ipdb.set_trace()
        #feature = self.basis_mat(feature)
        return feature
    
    

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):


        for l in range(self.level):
            for i in range(self.num_basises):
                cnt = l*self.num_basises + i
                #print(cnt, line[i].size())
                line[cnt] = torch.nn.Parameter(
                    F.interpolate(line[cnt].data, size=(int(res_target/(self.base**l)), 1), mode='bilinear', align_corners=True))
        for l in range(3):
            cnt = self.level*self.num_basises + l
            line[cnt] = torch.nn.Parameter(
                    F.interpolate(line[cnt].data, size=(res_target, 1), mode='bilinear', align_corners=True))  

        return line

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
        self.line = self.up_sampling_Vector(self.line, self.gridSize)       
        print(f'upsamping to {res_target}')


class TensorDVGO(TensorBase):
    def __init__(self,  gridSize, device, **kargs):
        super(TensorDVGO, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.feat_dim

    def init_svd_volume(self):
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.feat_dim, *self.gridSize.tolist()]).to(self.device))




    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):
        xyz_sampled = xyz_sampled.reshape(1,1,1,-1,3)
        return F.grid_sample(self.k0, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim,-1).T
    
    

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):

        pass

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.k0.data, size=tuple(self.gridSize.tolist()), mode='trilinear', align_corners=True))

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')



def zero_pads(data, voxel_size =16):
    if data.size(0)==1:
        data = data.squeeze(0)

    size = list(data.size())

    new_size = copy.deepcopy(size)
    for i in range(1,len(size)):
        if new_size[i]%voxel_size==0:
            continue
        new_size[i] = (new_size[i]//voxel_size+1)*voxel_size
    
    res= torch.zeros(new_size, device = data.device)
    res[:,:size[1],:size[2],:size[3]] = data.clone()
    return res

def zero_unpads(data, size):
    
    return data[:,:size[0],:size[1],:size[2]]


def split_volume(data, voxel_size =16):
    size = list(data.size())
    for i in range(1,len(size)):
        size[i] = size[i]//voxel_size

    res = []
    for x in range(size[1]):
        for y in range(size[2]):
            for z in range(size[3]):
                res.append(data[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size].clone())

    res = torch.stack(res)

    return res,size[1:]


def merge_volume(data,size):
    M, NF, Vx, Vy, Vz = data.shape  
    data_tmp = data[:size[0]*size[1]*size[2]].reshape(size[0],size[1],size[2],NF,Vx,Vy,Vz) 
    data_tmp = data_tmp.permute(3,0,4,1,5,2,6) 
    res = data_tmp.reshape(NF, size[0]*Vx, size[1]*Vy, size[2]*Vz) 
    return res  





class TensorPCA(TensorBase):

    def __init__(self,  gridSize, device, **kargs):
        super(TensorPCA, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.get_feature_size()

        self.merge_volume_cuda = Merge_Volume_CUDA()

        self.rec = None



        #self.basis_mat = torch.nn.Linear(self.num_componet*3*self.level, self.feat_dim, bias=False).to(device)
        
    @torch.no_grad()
    def eval(self):
        recon = self.U @ torch.diag(torch.exp(self.S)) 
        #h = recon.register_hook(lambda grad: print('recon',torch.abs(grad).sum()))
      
        recon = recon @ self.Vh

        
        recon = recon.reshape(recon.size(0),self.feat_dim,self.voxel_size,self.voxel_size,self.voxel_size)
      
    
        rec = zero_unpads(self.merge_volume_cuda(recon,self.grid_size), self.ori_size).unsqueeze(0)

        self.rec = rec

    @torch.no_grad()
    def train(self):
        self.rec = None

    def update_stepSize(self, gridSize):
        
        pass


    def init_svd_volume(self):

        assert self.cfg is not None, 'self.cfg is None!!!'
        self.voxel_size = self.cfg.pca_train.voxel_size
        self.ratio = self.cfg.pca_train.ratio

        threshold = self.cfg.pca_train.threshold
        

        path = os.path.join(self.cfg.basedir, self.cfg.expname)


       

        frames = self.cfg.pca_train.keyframes

        print('***********************************************')
        print('voxel_size:',self.voxel_size)
        print('ratio:',self.ratio)
        print('keyframes:',frames)

        big_data = []


        for frame_id in frames:
            print('process frame', frame_id)
            ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
            ckpt = torch.load(ckpt_path)
            model_states = ckpt['model_state_dict']

            self.ori_size = model_states['density'].size()[2:]
            self.gridSize =  torch.LongTensor(list(self.ori_size))

            density, self.grid_size = split_volume(zero_pads(model_states['density'].cpu(),voxel_size = self.voxel_size),voxel_size = self.voxel_size)
            
            k0, self.grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(),voxel_size = self.voxel_size),voxel_size = self.voxel_size)
            #print(model_states['xyz_min'])

            #density = density.permute(0,2,3,4,1)

            #ipdb.set_trace()
            #mask = torch.nn.functional.softplus(density-4.1).repeat(1,k0.size(1),1,1,1)

            #print(mask.max(), (mask<0.5).sum()/20,(mask>=0.5).sum()/20)
            #k0[mask<0.01] = -9

            big_data.append(torch.cat([density,k0], dim =1))


        big_data = torch.cat(big_data)
        big_data= big_data.reshape(big_data.size(0),big_data.size(1),-1)
        num_voxel_per_volume = big_data.size(0)//len(frames)

        assert big_data.size(1) == self.feat_dim + 1

        cnt_mask = big_data[:,0,:]
        cnt_mask = torch.nn.functional.softplus(cnt_mask-4.5) >0.2
        cnt_mask = cnt_mask.sum(dim=1)


        tmp_data = big_data[:,1:,:].cuda()
        tmp_data[cnt_mask<=threshold,:,:] = 0
        tmp_data = tmp_data.reshape(tmp_data.size(0),-1)
        U, S, Vh = torch.linalg.svd(tmp_data, full_matrices=False)

        cnt = S.size(0)
        length = int(self.ratio*cnt)
        #self.U = U[:num_voxel_per_volume,:length].contiguous()
        self.U = torch.nn.Parameter(U[:num_voxel_per_volume,:length].contiguous() )

        
        self.Vh = Vh[:length,:].contiguous()

        #self.Vh = torch.randn_like(Vh[:length,:], device = Vh.device)

        #ipdb.set_trace()

        self.S =  torch.nn.Parameter(3*torch.abs(torch.randn_like(S[:length],device = self.device)))

        print('empty ratio:', (1.0-(cnt_mask<=threshold).sum().item()/tmp_data.size(0))*100,'%')
        print('U size:',torch.prod(torch.tensor(self.U.size()))*4/1024/1024,'MB (float32)')
        print('S size:',torch.prod(torch.tensor(self.S.size()))*4/1024/1024,'MB (float32)')
        print('Vh size:',torch.prod(torch.tensor(self.Vh.size()))*4/1024/1024,'MB (float32)')

        print('***********************************************************')


    def get_feature_size(self):
        return self.feat_dim

    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):

        
        #h = self.S.register_hook(lambda grad: print('S',torch.abs(grad).sum()))
        if self.rec is None:


            recon = self.U @ torch.diag(torch.exp(self.S)) 
            #h = recon.register_hook(lambda grad: print('recon',torch.abs(grad).sum()))
        
            recon = recon @ self.Vh

            
            recon = recon.reshape(recon.size(0),self.feat_dim,self.voxel_size,self.voxel_size,self.voxel_size)

        
            rec = zero_unpads(self.merge_volume_cuda(recon,self.grid_size), self.ori_size).unsqueeze(0)
        else:
            rec = self.rec.clone()
        #rec = zero_unpads(merge_volume(recon,self.grid_size), self.ori_size).unsqueeze(0)



        
        xyz_sampled = xyz_sampled.reshape(1,1,1,-1,3)

        ret = F.grid_sample(rec, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim,-1).T
     
        return ret
          


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        pass
        print(f'Tensor PCA')




class TensorDCT(TensorBase):

    def __init__(self,  gridSize, device, **kargs):
        super(TensorDCT, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.get_feature_size()

        self.merge_volume_cuda = Merge_Volume_CUDA()

        self.rec = None



        #self.basis_mat = torch.nn.Linear(self.num_componet*3*self.level, self.feat_dim, bias=False).to(device)
        
    @torch.no_grad()
    def eval(self):
        recon = self.U @ self.Vh
        
        recon = recon.reshape(recon.size(0),self.feat_dim,self.voxel_size,self.voxel_size,self.voxel_size)
      
    
        rec = zero_unpads(self.merge_volume_cuda(recon,self.grid_size), self.ori_size).unsqueeze(0)

        self.rec = rec

    @torch.no_grad()
    def train(self):
        self.rec = None

    def update_stepSize(self, gridSize):
        
        pass


    def init_svd_volume(self):

        assert self.cfg is not None, 'self.cfg is None!!!'
        self.voxel_size = self.cfg.pca_train.voxel_size
        self.ratio = self.cfg.pca_train.ratio

    
        

        path = os.path.join(self.cfg.basedir, self.cfg.expname)


       

        frames = self.cfg.pca_train.keyframes

        print('***********************************************')
        print('voxel_size:',self.voxel_size)
        print('ratio:',self.ratio)
        print('keyframes:',frames)

        big_data = []


        frame_id = frames[0]


      
        print('process frame', frame_id)
        ckpt_path = os.path.join(path, 'fine_last_%d.tar' % frame_id)
        ckpt = torch.load(ckpt_path)
        model_states = ckpt['model_state_dict']

        self.ori_size = model_states['density'].size()[2:]
        self.gridSize =  torch.LongTensor(list(self.ori_size))

        density, self.grid_size = split_volume(zero_pads(model_states['density'].cpu(),voxel_size = self.voxel_size),voxel_size = self.voxel_size)
        
        k0, self.grid_size = split_volume(zero_pads(model_states['k0.k0'].cpu(),voxel_size = self.voxel_size),voxel_size = self.voxel_size)
        #print(model_states['xyz_min'])

        #density = density.permute(0,2,3,4,1)

        #ipdb.set_trace()
        #mask = torch.nn.functional.softplus(density-4.1).repeat(1,k0.size(1),1,1,1)

        #print(mask.max(), (mask<0.5).sum()/20,(mask>=0.5).sum()/20)
        #k0[mask<0.01] = -9

        big_data.append(torch.cat([density,k0], dim =1))


        big_data = torch.cat(big_data)
        big_data= big_data.reshape(big_data.size(0),big_data.size(1),-1)

        assert big_data.size(1) == self.feat_dim + 1

     


        tmp_data = big_data[:,1:,:].cuda()
        tmp_data = tmp_data.reshape(tmp_data.size(0),-1)

        x = torch.tensor(np.arange(self.voxel_size))
        fx = torch.tensor(np.arange(self.feat_dim))
        grid_feat, grid_x,grid_y,grid_z = torch.meshgrid([fx,x,x,x],indexing='ij')
        
        p = self.ratio**(1/4)
        pi=3.141592653

        Vh= []

        sf = 6


        for kf in range(0,sf):
            for k1 in range(0,sf):
                for k2 in range(0,sf):
                    for k3 in range(0,sf):
                        res = torch.cos(pi*(grid_feat+0.5)*kf/self.feat_dim)*torch.cos(pi*(grid_x+0.5)*k1/self.voxel_size)*torch.cos(pi*(grid_y+0.5)*k2/self.voxel_size)*torch.cos(pi*(grid_z+0.5)*k3/self.voxel_size)
                        Vh.append(res.reshape(-1))

        for kf in range(sf,self.feat_dim, 1):
            for k1 in range(sf,self.voxel_size, 2):
                for k2 in range(sf,self.voxel_size, 2):
                    for k3 in range(sf,self.voxel_size, 2):
                        res = torch.cos(pi*(grid_feat+0.5)*kf/self.feat_dim)*torch.cos(pi*(grid_x+0.5)*k1/self.voxel_size)*torch.cos(pi*(grid_y+0.5)*k2/self.voxel_size)*torch.cos(pi*(grid_z+0.5)*k3/self.voxel_size)
                        Vh.append(res.reshape(-1))
        Vh = torch.stack(Vh)

        assert Vh.size(1) == tmp_data.size(1)


        self.U = torch.nn.Parameter(torch.randn(tmp_data.size(0), Vh.size(0),device = tmp_data.device) )
        self.Vh = Vh.to(tmp_data.device)
        #self.Vh = torch.randn_like(Vh[:length,:], device = Vh.device)

        #ipdb.set_trace()

        
        print('U size:',torch.prod(torch.tensor(self.U.size()))*4/1024/1024,'MB (float32)', self.U.size() )
        print('Vh size:',torch.prod(torch.tensor(self.Vh.size()))*4/1024/1024,'MB (float32)', self.Vh.size())

        print('***********************************************************')


    def get_feature_size(self):
        return self.feat_dim

    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):

        
        #h = self.S.register_hook(lambda grad: print('S',torch.abs(grad).sum()))
        if self.rec is None:


            recon = self.U @ self.Vh
            
            recon = recon.reshape(recon.size(0),self.feat_dim,self.voxel_size,self.voxel_size,self.voxel_size)

        
            rec = zero_unpads(self.merge_volume_cuda(recon,self.grid_size), self.ori_size).unsqueeze(0)
        else:
            rec = self.rec.clone()
        #rec = zero_unpads(merge_volume(recon,self.grid_size), self.ori_size).unsqueeze(0)



        
        xyz_sampled = xyz_sampled.reshape(1,1,1,-1,3)

        ret = F.grid_sample(rec, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim,-1).T
     
        return ret
          


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        pass
        print(f'Tensor DCT')





class TensorDVGO(TensorBase):
    def __init__(self,  gridSize, device, **kargs):
        super(TensorDVGO, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.feat_dim

    def init_svd_volume(self):
        self.k0 = torch.nn.Parameter(torch.empty([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)
        std = 1e-4
        self.k0.data.uniform_(-std, std)

    # xyz_sampled: (..., 3 )
    def compute_features(self, xyz_sampled):
        xyz_sampled = xyz_sampled.reshape(1,1,1,-1,3)
        return F.grid_sample(self.k0, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim,-1).T
    
    

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):

        pass

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.k0.data, size=tuple(self.gridSize.tolist()), mode='trilinear', align_corners=True))

        
        print(f'upsamping to {res_target}')

class TensorDVGORes(TensorBase):
    '''
    这里k0 表示的是TensorDVGO两帧k0的差
    '''
    def __init__(self,  gridSize, device, **kargs):
        super(TensorDVGORes, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.feat_dim
        print("---------------init TensorDVGORes --------------")

    def init_svd_volume(self):
        self.k0 = torch.nn.Parameter(torch.empty([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)
        std = 1e-4
        self.k0.data.uniform_(-std, std)
        former_k0 = torch.tensor(torch.zeros([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)
        self.register_buffer('former_k0',former_k0)
        self.former_k0_cur = torch.tensor(torch.zeros([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)

    def compute_features(self, xyz_sampled):
        xyz_sampled = xyz_sampled.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.former_k0_cur+self.k0, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim, -1).T

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):
        pass


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
        self.former_k0_cur= torch.tensor(
            F.interpolate(self.former_k0, size=tuple(self.gridSize.tolist()), mode='trilinear', align_corners=True))
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.k0.data, size=tuple(self.gridSize.tolist()), mode='trilinear', align_corners=True))

        print(f'upsamping to {res_target}')

class TensorDVGODeform(TensorBase):
    '''
    former k0 是前一帧全分辨率的k0， k0=former_k0_cur
    '''
    def __init__(self,  gridSize, device, **kargs):
        super(TensorDVGODeform, self).__init__(gridSize, device, **kargs)
        self.real_dim = self.feat_dim
        print("---------------init TensorDVGORes --------------")

    def init_svd_volume(self):
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)
        self.former_k0 = torch.tensor(torch.zeros([1, self.feat_dim, *self.gridSize.tolist()])).to(self.device)


    def compute_features(self, xyz_sampled):
        xyz_sampled = xyz_sampled.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.k0, xyz_sampled, mode='bilinear', align_corners=True).reshape(self.feat_dim, -1).T

    @torch.no_grad()
    def up_sampling_Vector(self, line, res_target):
        pass

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.former_k0.data, size=tuple(self.gridSize.tolist()), mode='trilinear', align_corners=True))

        print(f'upsamping to {res_target}')

if __name__ == "__main__":

    a = TensorPCA((150,350,150),'cuda', feat_dim = 17)
    pts = torch.rand(5,3)
    feats = a.compute_features(pts)

    a.upsample_volume_grid((50,440,256))



    print(fibonacci_sphere(5))
    ipdb.set_trace()
    

