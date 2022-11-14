import time
import torch

class Timer(object):
    def __init__(self,name,device = 'cuda:0'):
        self.name=name
        self.device=device
    def __enter__(self):
        torch.cuda.synchronize(device=self.device)
        self.time_start=time.time()
    def __exit__(self,exc_type,exc_value,traceback):
        torch.cuda.synchronize(device=self.device)
        print( self.name, time.time() - self.time_start, 's')


from ._dct import dct, idct, dct1, idct1, dct_2d, idct_2d, dct_3d, idct_3d, LinearDCT, apply_linear_2d, apply_linear_3d

from .utils import *


from .metrics import *

from .huffmanencode import huffman_encode

from .quant import *

from .encoder_jpeg import *

from .motion import *
from .compress_utils import *

