from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import sys
import os
import torch
from prototype_jpeg.utils import (rgb2ycbcr, ycbcr2rgb, downsample, upsample, block_slice,
                    block_combine, dct2d, idct2d, quantize, Y, CB, CR)


def jpeg_stoch(data, dataname):
    data = np.asarray(data)
    data=np.rint(data)
    tmp = torch.Tensor(data)
    print(f"zero percent {dataname}: ", torch.sum(tmp == 0) / tmp.nelement())


grey_level=False
subsampling_mode=1

qualities = (90, 80, 50, 20, 10, 5)


(_, axarr) = plt.subplots(5, len(qualities),figsize=(20, 20) )

imgname = '1'
srcFileName=f'/data/new_disk2/wangla/Projects/7.29jpeg_expr/{imgname}.bmp'
srcImage = Image.open(srcFileName)
img_arr= np.asarray(srcImage)

for j, q in enumerate(qualities):
    vis_key=Y
    dct=[]
    quant=[]
    DC=[]
    dDC=[]
    #heat=[]
    AC=[]

    if grey_level:
        data = {Y: img_arr.astype(float)}

    else:  # RGB
        # Color Space Conversion (w/o Level Offset)
        data = rgb2ycbcr(*(img_arr[:, :, idx] for idx in range(3)))

        # Subsampling
        data[CB] = downsample(data[CB], subsampling_mode)
        data[CR] = downsample(data[CR], subsampling_mode)

        data[Y] = data[Y] - 128


        for key, layer in data.items():
            if key != vis_key:
                continue

            nrows, ncols = layer.shape

            # Pad Layers to 8N * 8N
            data[key] = np.pad(
                layer,
                (
                    (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
                    (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
                ),
                mode='constant'
            )

            # Block Slicing
            data[key] = block_slice(data[key], 8, 8)

            for idx, block in enumerate(data[key]):
                # 2D DCT
                data[key][idx] = dct2d(block)

                dct.append(data[key][idx].copy())
                # Quantization
                data[key][idx] = quantize(data[key][idx], key, quality=q)

                quant.append(data[key][idx].copy())
                DC.append(data[key][idx][0])
                AC.append(data[key][idx][1:])

                if idx==0:
                    dDC.append(DC[-1])
                else:
                    dDC.append(DC[-1]-DC[-2])


            axarr[0][j].hist(np.rint(np.asarray(dct)).ravel(), log=True)
            jpeg_stoch(dct,'dct_'+str(q))
            axarr[1][j].hist(np.rint(np.asarray(quant)).ravel(), log=True)
            jpeg_stoch(quant, 'quant_'+str(q))
            axarr[2][j].hist(np.rint(np.asarray(DC)).ravel(), log=True)
            jpeg_stoch(DC, 'DC_'+str(q))
            axarr[3][j].hist(np.rint(np.asarray(dDC)).ravel(), log=True)
            jpeg_stoch(dDC, 'dDC_'+str(q))
            axarr[4][j].hist(np.rint(np.asarray(AC)).ravel(), log=True)
            jpeg_stoch(AC, 'AC_'+str(q))

            data[key] = np.rint(data[key]).astype(int)



# plt.figure(figsize=(12, 12))
plt.savefig('/data/new_disk2/wangla/Projects/7.29jpeg_expr/1_Y.png')
plt.close()


