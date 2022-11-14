import os
import torch
from torch.utils.cpp_extension import load
import torch.nn as nn
import ipdb

parent_dir = os.path.dirname(os.path.abspath(__file__))
sources=['cuda/voxel_utils.cpp', 'cuda/voxel_utils.cu']
voxel_utils_cuda = load(
        name='voxel_utils_cuda',
        sources=[os.path.join(parent_dir, path) for path in sources],
        verbose=True)


def merge_volume(data,size):
    M, NF, Vx, Vy, Vz = data.shape  
    data_tmp = data[:size[0]*size[1]*size[2]].reshape(size[0],size[1],size[2],NF,Vx,Vy,Vz) 
    data_tmp = data_tmp.permute(3,0,4,1,5,2,6) 
    res = data_tmp.reshape(NF, size[0]*Vx, size[1]*Vy, size[2]*Vz) 
    return res  


class Merge_Volume_OP_CUDA(torch.autograd.Function):

    # in_tensor:  (N,dim,voxel_size,voxel_size,voxel_size)
    # in_size :  (3,)
    @staticmethod
    def forward(ctx, in_tensor, in_size):  
        with torch.no_grad():
            in_size = torch.tensor(in_size,device ='cpu').long()
            voxel_size = in_tensor.size(-1)
            feat_dim = in_tensor.size(1)
            ret  = in_tensor.permute(0,2,3,4,1).contiguous() 

            out_tensor = torch.zeros(in_size[0]*voxel_size,in_size[1]*voxel_size,in_size[2]*voxel_size,feat_dim,device=in_tensor.device)



            voxel_utils_cuda.merge_volume_forward(ret,in_size,out_tensor)


            out_tensor = out_tensor.permute(3,0,1,2)


        ctx.save_for_backward(in_tensor, in_size)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_tensor):
        in_tensor, in_size = ctx.saved_tensors

        with torch.no_grad():

            ret = grad_tensor.permute(1,2,3,0).contiguous() 
            voxel_size = in_tensor.size(-1)
            feat_dim = in_tensor.size(1)


            out_grad_tensor = torch.zeros(in_tensor.size(0),voxel_size,voxel_size,voxel_size,feat_dim,device =grad_tensor.device )
        
            voxel_utils_cuda.merge_volume_backward(ret,in_size,out_grad_tensor)

            out_grad_tensor = out_grad_tensor.permute(0,4,1,2,3)
    

        return out_grad_tensor, None 



class Merge_Volume_CUDA(nn.Module):
    def __init__(self):
        super(Merge_Volume_CUDA, self).__init__()

    def forward(self, in_tensor, in_size):
        return Merge_Volume_OP_CUDA.apply(in_tensor, in_size)



if __name__ == "__main__":
    import time
    size = torch.tensor([10,36,12]).long()
    N = 10*36*12
    feat_dim = 17
    voxel_size = 10

    in_tensor= nn.Parameter(torch.randn(N,feat_dim,voxel_size,voxel_size,voxel_size).float().cuda())
    #in_tensor.retain_grad()

    out_tensor_ori = merge_volume(in_tensor,size)

    model = Merge_Volume_CUDA()


    torch.cuda.synchronize()
    beg = time.time()

    for i in range(100):
        out_tensor = model(in_tensor,size)

    torch.cuda.synchronize()
    print('time:', time.time() - beg)


    #print(out_tensor_ori)
    #print(out_tensor)
    print((out_tensor_ori-out_tensor).abs().sum())


    tt = torch.randn_like(out_tensor_ori).cuda()

    out_tensor_ori.backward(tt)

    grad1 = in_tensor.grad.clone()
    in_tensor.grad = None
    out_tensor.backward(tt)
    print((in_tensor.grad-grad1).abs().sum())

