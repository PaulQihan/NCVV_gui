from PIL import Image
from scipy import fftpack
import numpy
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import sys
import torch

import os

import seaborn as sns

zigzagOrder = numpy.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])
#std_quant_tbl from libjpeg::jcparam.c


std_luminance_quant_tbl = numpy.array(
[ 16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99],dtype=int)
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8,8])

std_chrominance_quant_tbl = numpy.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8,8])

def main():

    # inputBMPFileName outputJPEGFilename quality(from 1 to 100) DEBUGMODE(0 or 1)
    # example:
    # ./lena.bmp ./output.jpg 80 0

    if(len(sys.argv)!=5):
        print('inputBMPFileName outputJPEGFilename quality(from 1 to 100) DEBUGMODE(0 or 1)')
        print('example:')
        print('./lena.bmp ./output.jpg 80 0')
        return

    imgname = sys.argv[1]
    srcFileName=f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}.bmp'
    outputJPEGFileName = sys.argv[2]
    quality = float(sys.argv[3])
    DEBUG_MODE = int(sys.argv[4])


    numpy.set_printoptions(threshold=numpy.inf)
    srcImage = Image.open(srcFileName)
    srcImageWidth, srcImageHeight = srcImage.size
    print('srcImageWidth = %d srcImageHeight = %d' % (srcImageWidth, srcImageHeight))
    print('srcImage info:\n', srcImage)
    srcImageMatrix = numpy.asarray(srcImage)

    imageWidth = srcImageWidth
    imageHeight = srcImageHeight
    # add width and height to %8==0
    if (srcImageWidth % 8 != 0):
        imageWidth = srcImageWidth // 8 * 8 + 8
    if (srcImageHeight % 8 != 0):
        imageHeight = srcImageHeight // 8 * 8 + 8

    print('added to: ', imageWidth, imageHeight)

    # copy data from srcImageMatrix to addedImageMatrix
    addedImageMatrix = numpy.zeros((imageHeight, imageWidth, 3), dtype=numpy.uint8)
    for y in range(srcImageHeight):
        for x in range(srcImageWidth):
            addedImageMatrix[y][x] = srcImageMatrix[y][x]


    # split y u v
    yImage,uImage,vImage = Image.fromarray(addedImageMatrix).convert('YCbCr').split()

    yImageMatrix = numpy.asarray(yImage).astype(int)
    uImageMatrix = numpy.asarray(uImage).astype(int)
    vImageMatrix = numpy.asarray(vImage).astype(int)
    if(DEBUG_MODE==1):
        print('yImageMatrix:\n', yImageMatrix)
        print('uImageMatrix:\n', uImageMatrix)
        print('vImageMatrix:\n', vImageMatrix)


    yImageMatrix = yImageMatrix - 127
    uImageMatrix = uImageMatrix - 127
    vImageMatrix = vImageMatrix - 127


    if(quality <= 0):
        quality = 1
    if(quality > 100):
        quality = 100
    if(quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    luminanceQuantTbl = numpy.array(numpy.floor((std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl[luminanceQuantTbl > 255] = 255
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)
    print('luminanceQuantTbl:\n', luminanceQuantTbl)
    chrominanceQuantTbl = numpy.array(numpy.floor((std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    print('chrominanceQuantTbl:\n', chrominanceQuantTbl)
    blockSum = imageWidth // 8 * imageHeight // 8

    yDC = numpy.zeros([blockSum], dtype=int)
    uDC = numpy.zeros([blockSum], dtype=int)
    vDC = numpy.zeros([blockSum], dtype=int)
    dyDC = numpy.zeros([blockSum], dtype=int)
    duDC = numpy.zeros([blockSum], dtype=int)
    dvDC = numpy.zeros([blockSum], dtype=int)

    print('blockSum = ', blockSum)

    tmpy=numpy.zeros_like(yImageMatrix)
    tmpu=numpy.zeros_like(uImageMatrix)
    tmpv=numpy.zeros_like(vImageMatrix)

    y_heat=np.zeros([8,8])
    u_heat=np.zeros([8,8])
    v_heat=np.zeros([8,8])

    yAC=[]
    uAC=[]
    vAC=[]

    blockNum = 0
    for y in range(0, imageHeight, 8):
        for x in range(0, imageWidth, 8):
            #print('block (y,x): ',y, x, ' -> ', y + 8, x + 8)
            yDctMatrix = fftpack.dct(fftpack.dct(yImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            uDctMatrix = fftpack.dct(fftpack.dct(uImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            vDctMatrix = fftpack.dct(fftpack.dct(vImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            if(blockSum<=8):
                print('yDctMatrix:\n',yDctMatrix)
                print('uDctMatrix:\n',uDctMatrix)
                print('vDctMatrix:\n',vDctMatrix)

            yQuantMatrix = numpy.rint(yDctMatrix / luminanceQuantTbl)
            uQuantMatrix = numpy.rint(uDctMatrix / chrominanceQuantTbl)
            vQuantMatrix = numpy.rint(vDctMatrix / chrominanceQuantTbl)
            if(DEBUG_MODE==1):
                print('yQuantMatrix:\n',yQuantMatrix)
                print('uQuantMatrix:\n',uQuantMatrix)
                print('vQuantMatrix:\n',vQuantMatrix)

            tmpy[y:y + 8, x:x + 8]=yQuantMatrix
            tmpu[y:y + 8, x:x + 8]=uQuantMatrix
            tmpv[y:y + 8, x:x + 8]=vQuantMatrix
            y_heat+=yQuantMatrix
            u_heat+=uQuantMatrix
            v_heat+=vQuantMatrix

            yZCode = yQuantMatrix.reshape([64])[zigzagOrder]
            uZCode = uQuantMatrix.reshape([64])[zigzagOrder]
            vZCode = vQuantMatrix.reshape([64])[zigzagOrder]
            yZCode = yZCode.astype(numpy.int)
            uZCode = uZCode.astype(numpy.int)
            vZCode = vZCode.astype(numpy.int)

            yDC[blockNum] = yZCode[0]
            uDC[blockNum] = uZCode[0]
            vDC[blockNum] = vZCode[0]

            if (blockNum == 0):
                dyDC[blockNum] = yDC[blockNum]
                duDC[blockNum] = uDC[blockNum]
                dvDC[blockNum] = vDC[blockNum]
            else:
                dyDC[blockNum] = yDC[blockNum] - yDC[blockNum - 1]
                duDC[blockNum] = uDC[blockNum] - uDC[blockNum - 1]
                dvDC[blockNum] = vDC[blockNum] - vDC[blockNum - 1]

            yAC.append(yZCode[1:])
            uAC.append(uZCode[1:])
            vAC.append(vZCode[1:])

            blockNum = blockNum + 1

    y_heat/=blockSum
    u_heat/=blockSum
    v_heat/=blockSum

    jpeg_heatmap(y_heat, 'ydct', imgname, quality)
    jpeg_heatmap(u_heat, 'udct', imgname, quality)
    jpeg_heatmap(v_heat, 'vdct', imgname, quality)

    jpeg_stoch(tmpy,'ydct',imgname,quality)
    jpeg_stoch(tmpu,'udct',imgname,quality)
    jpeg_stoch(tmpv,'vdct',imgname,quality)

    jpeg_stoch(yDC, 'yDC', imgname, quality)
    jpeg_stoch(uDC, 'uDC', imgname, quality)
    jpeg_stoch(vDC, 'vDC', imgname, quality)

    jpeg_stoch(dyDC[1:], 'dyDC', imgname, quality)
    jpeg_stoch(duDC[1:], 'duDC', imgname, quality)
    jpeg_stoch(dvDC[1:], 'dvDC', imgname, quality)

    yAC=np.asarray(yAC)
    uAC=np.asarray(uAC)
    vAC=np.asarray(vAC)

    jpeg_stoch(yAC[1:], 'yAC', imgname, quality)
    jpeg_stoch(uAC[1:], 'uAC', imgname, quality)
    jpeg_stoch(vAC[1:], 'vAC', imgname, quality)



def jpeg_stoch(data,dataname,imgname,quality):
    os.makedirs(f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}',exist_ok=True)
    plt.hist(data.ravel(), log=True)
    plt.savefig(f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}/{dataname}_{quality}.png')
    plt.close()

    tmp = torch.Tensor(data)
    print(f"zero percent {dataname}: ", torch.sum(tmp == 0) / tmp.nelement())

def jpeg_heatmap(data,dataname,imgname,quality):
    os.makedirs(f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}', exist_ok=True)
    sns.heatmap(data, annot=True)
    plt.savefig(f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}/heatmap_{dataname}_{quality}.png')
    plt.close()





if __name__ == '__main__':
    main()
