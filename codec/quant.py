import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from codec import dct_3d, idct_3d,huffman_encode

def get_3d_info(b):
    #https://blog.csdn.net/weixin_44586750/article/details/111667510
    a=np.array([1/math.sqrt(3),1/math.sqrt(3),1/math.sqrt(3)])
    p=a*(np.dot(a.T,b)/np.dot(a.T,a))
    e=b-p
    return np.linalg.norm(p),np.linalg.norm(e)

def linear2dinter(p,tbl):
    x,y=p[0],p[1]
    if x>7:
        tmp=x-7
        x=7
        y+=tmp

    if y>7:
        tmp=y-7
        y=7
        x+=tmp

    xc,yc=int(x),int(y)
    xf,yf=x-xc,y-yc

    return tbl[xc,yc]*(1-xf)*(1-yf)+tbl[min(7,xc+1),yc]*(xf)*(1-yf)+tbl[xc,min(7,yc+1)]*(1-xf)*(yf)+tbl[min(7,xc+1),min(7,yc+1)]*(xf)*(yf)



def get_3d_tbl_ele(p3d,tbl):
    p3d = np.array(p3d)
    d3d,e3d=get_3d_info(p3d)
    d2d,e2d=d3d/math.sqrt(3)*math.sqrt(2), e3d/math.sqrt(3)*math.sqrt(2)
    if e2d> d2d:
        e2d,d2d=d2d,e2d # maybe not right
    p1=d2d*np.array([1/math.sqrt(2),1/math.sqrt(2)])+e2d*np.array([1/math.sqrt(2),-1/math.sqrt(2)])
    p2=d2d*np.array([1/math.sqrt(2),1/math.sqrt(2)])+e2d*np.array([-1/math.sqrt(2),1/math.sqrt(2)])
    # tmp1=linear2dinter(p1,tbl)
    # tmp2=linear2dinter(p2,tbl)
    return (linear2dinter(p1,tbl)+linear2dinter(p2,tbl))/2

def gen_3d_quant_tbl(constant=1):
    QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],     # luminance quantization table
                    [12, 12, 14, 19, 26, 48, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])
    #QTY[QTY<56]=1
    # QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],     # luminance quantization table
    #                 [12, 12, 14, 19, 26, 48, 60, 55],
    #                 [14, 13, 16, 24, 40, 56, 56, 56],
    #                 [14, 17, 22, 29, 51, 56, 56, 56],
    #                 [18, 22, 37, 56, 56, 56, 56, 56],
    #                 [24, 35, 55, 64, 56, 56, 56, 56],
    #                 [49, 64, 78, 87, 56, 56, 56, 56],
    #                 [72, 92, 95, 98, 56, 56, 56, 56]])
    QTY_3d=np.zeros((8,8,8)) #

    for x in range(8):
        for y  in range(8):
            for z in range(8):
                QTY_3d[x,y,z]=constant*get_3d_tbl_ele([x,y,z],QTY)
                #QTY_3d[x,y,z]=constant*(QTY[x,y]+QTY[y,z]+QTY[z,x])/3 #不太对

    #print(QTY_3d)
    return torch.Tensor(QTY_3d).cuda()

def vis_quant_bin(RES,file_path=''):
    print("saving hist image ", file_path)
    raw = RES.cpu().numpy()
    plt.hist(raw.ravel(), log=True)
    plt.savefig(file_path)
    plt.close()

def quant_encode(RES,table,quant_num=1,vis_path='',debug=False):

    RES = RES / (table.unsqueeze(0).unsqueeze(0)*quant_num)
    R_min=RES.min()
    R_max=RES.max()
    #if debug:
    print("min", RES.min())
    print("max", RES.max())
    print("zero percent: ", torch.sum(torch.floor(RES)==0)/RES.nelement())
    # naive quant do not know right or not, normalize to 0-255
    vis_quant_bin(RES,
                  file_path=f'/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_tabtest_{quant_num}.png')
    RES = torch.round(RES) #需要测试一下
    # RES=torch.round(RES) 为什么用round 效果反而不好？

    return RES,R_min,R_max

def quant_norm_uint8_encode(RES,table,vis_path='',debug=False):
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    #if debug:
    print("min", RES.min())
    print("max", RES.max())
    print("zero percent: ", torch.sum(torch.floor(RES)==0)/RES.nelement())
    # naive quant do not know right or not, normalize to 0-255
    RES = (RES - RES.min()) / (RES.max() - RES.min()) * 255
    RES = torch.round(RES)
    # RES=torch.round(RES) 为什么用round 效果反而不好？
    RES = RES.to(torch.uint8)

    return RES,R_min,R_max

def quant_norm_uint12_encode(RES,table,vis_path='',debug=False):
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    #if debug:
    print("min", RES.min())
    print("max", RES.max())
    print("zero percent: ", torch.sum(torch.floor(RES)==0)/RES.nelement())

    vis_quant_bin(RES,
                  file_path='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_1_int12.png')


    RES = (RES - RES.min()) / (RES.max() - RES.min()) * 4095
    RES = torch.round(RES)
    # RES=torch.round(RES) 为什么用round 效果反而不好？
    RES = RES.to(torch.int16)

    return RES,R_min,R_max

def quant_decode(RES,R_min,R_max,table,quant_num=1,vis_path='',debug=False):

    RES = RES * table.unsqueeze(0).unsqueeze(0)*quant_num
    return RES

def quant_norm_uint8_decode(RES,R_min,R_max,table,vis_path='',debug=False):

    RES=RES.to(torch.float)
    RES=RES/255* (R_max - R_min) + R_min
    RES = RES * table.unsqueeze(0).unsqueeze(0)

    return RES

def quant_norm_uint12_decode(RES,R_min,R_max,table,vis_path='',debug=False):

    RES=RES.to(torch.float)
    RES=RES/4095* (R_max - R_min) + R_min
    RES = RES * table.unsqueeze(0).unsqueeze(0)

    return RES


def quant_norm(RES,table,vis_path=''):
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    print("min", RES.min())
    print("max", RES.max())
    # naive quant do not know right or not, normalize to 0-255
    RES = (RES - RES.min()) / (RES.max() - RES.min()) * 255
    # RES=torch.round(RES) 为什么用round 效果反而不好？
    RES=RES.to(torch.uint8)
    if vis_path != '':
        vis_quant_bin(RES, vis_path)
    RES = RES.to(torch.float)
    RES=RES/255*(R_max-R_min)+R_min
    RES=RES*table.unsqueeze(0).unsqueeze(0)

    return RES

def quant_norm_int16(RES,table,vis_path=''):
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    print("min", RES.min())
    print("max", RES.max())
    # naive quant do not know right or not, normalize to 0-255
    RES = ((RES - RES.min()) / (RES.max() - RES.min())-0.5)*2 * 32767
    # RES=torch.round(RES) 为什么用round 效果反而不好？
    RES=RES.to(torch.int16)
    if vis_path != '':
        vis_quant_bin(RES, vis_path)
    RES = RES.to(torch.float)
    RES=(RES/(2 * 32767)+0.5)*(R_max-R_min)+R_min
    RES=RES*table.unsqueeze(0).unsqueeze(0)

    return RES

def quant_norm_int12(RES,table,vis_path=''):
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    print("min", RES.min())
    print("max", RES.max())
    # naive quant do not know right or not, normalize to 0-255
    RES = (RES - RES.min()) / (RES.max() - RES.min()) * 4095
    RES=torch.round(RES) #为什么用round 效果反而不好？
    RES = RES.to(torch.int16)
    if vis_path != '':
        vis_quant_bin(RES, vis_path)
    RES = RES.to(torch.float)
    RES = RES / 4095 * (R_max - R_min) + R_min
    RES=RES*table.unsqueeze(0).unsqueeze(0)

    return RES


quant_encoders={
    "jpeg": quant_encode,
    "uint8": quant_norm_uint8_encode,
    "uint12": quant_norm_uint12_encode,
}

quant_decoders={
    "jpeg": quant_decode,
    "uint8": quant_norm_uint8_decode,
    "uint12": quant_norm_uint12_decode,
}


def encode_hqh(data,table,path,quant_type='uint12',quant_const=1):
    '''
    先dct，量化 然后 normalization 到 Huffman表统计，编码
    :return:
    '''

    os.makedirs(path, exist_ok=True)
    quant_func=quant_encoders[quant_type]

    RES = dct_3d(data, norm = 'ortho')
    print("dct min", RES.min())
    print("dct max", RES.max())
    print("dct zero percent: ", torch.sum(torch.round(RES) == 0) / RES.nelement())

    RES[:, :1], RES_density_min, RES_density_max = quant_func(RES[:, :1], table,quant_const)
    RES[:, 1:], RES_color_min, RES_color_max = quant_func(RES[:, 1:], table,quant_const)
    # vis_quant_bin(RES[:, :1], file_path='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_density_int8.png')
    # vis_quant_bin(RES[:, 1:],
    #               file_path='/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/quant_color_int8.png')

    huffman_encode(RES[:, :1], path + '/density')
    huffman_encode(RES[:, 1:4], path + '/color1')
    huffman_encode(RES[:, 4:7], path + '/color2')
    huffman_encode(RES[:, 7:10], path + '/color3')
    huffman_encode(RES[:, 10:13], path + '/color4')

    return RES_density_min, RES_density_max , RES_color_min, RES_color_max,RES

def decode_hqh(RES,RES_density_min, RES_density_max , RES_color_min, RES_color_max,table,path,quant_type='uint12',quant_const=1):

    #huffman_denode not implement

    quant_func = quant_decoders[quant_type]
    RES[:, :1]=quant_func(RES[:, :1],RES_density_min,RES_density_max,table,quant_const)
    RES[:, 1:] = quant_func(RES[:, 1:], RES_color_min, RES_color_max, table,quant_const)

    res= idct_3d(RES, norm = 'ortho')

    return res


def encode_jpeg_dc(data,table,path):
    '''
        先dct，量化 然后用 DPCM on DC Components“
        :return:
        '''
    RES = dct_3d(data, norm = 'ortho')
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    RES = RES.to(torch.int16)

    return RES


#useless

def encode_int8_v0(data,table,vis_path=''):
    '''contains dct quant'''
    #normalize 到（-128-127）
    d_min = data.min()
    d_max = data.max()
    print("d_min", d_min)
    print("d_max", d_max)
    data = (data - d_min) / (d_max - d_min) * 255 -128
    print(data.min())
    print(data.max())
    RES = dct_3d(data)  # torch.Size([1134, 13, 16, 16, 16])
    print("min", RES.min())
    print("max", RES.max())
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min = RES.min()
    R_max = RES.max()
    print("min", RES.min())
    print("max", RES.max())

    RES=RES.to(torch.int16)
    return RES,d_min,d_max

def encode_int8(data,table,vis_path=''):
    '''contains dct quant'''
    #normalize 到（-128-127）

    RES = dct_3d(data, norm = 'ortho')  # torch.Size([1134, 13, 16, 16, 16])
    vis_quant_bin(RES, "/data/new_disk2/wangla/tmp/NCVV/logs/NHR/xzq_white_frames/dct_hist_color_norm.png")
    d_min = data.min()
    d_max = data.max()

    print("min", RES.min())
    print("max", RES.max())

    RES=RES.to(torch.int16)
    return RES,d_min,d_max
#visual

def quant(RES,table,vis_path=''): #no norm
    RES = RES / table.unsqueeze(0).unsqueeze(0)
    R_min=RES.min()
    R_max=RES.max()
    print("min", RES.min())
    print("max", RES.max())

    if vis_path != '':
        vis_quant_bin(RES, vis_path)
    RES=RES*table.unsqueeze(0).unsqueeze(0)

    return RES


def quantize_quality(table,  quality):
    if quality >= 100:
        return torch.ones_like(table)
    factor = 5000 / quality if quality < 50 else 200 - 2 * quality
    return table * factor / 100

def anal_res(RES):
    # for i in range(13):
    #     print(" vis min", RES[i].min())
    #     print(" vis max", RES[i].max())
    #print(RES[23,1])
    # tmp=RES.reshape((-1,512))
    # a_max=torch.argmax(tmp,dim=1)
    # a_min=torch.argmin(tmp,dim=1)
    # for i in range(a_min.size(0)):
    #     #print(a_max[i].item(),end=" ")
    #     print(a_min[i].item(), end=" ")
    pass




