import numpy as np
import os
import json


root_path = '..'

with open(os.path.join(root_path,'run_voxel_sample.sh'), 'w', encoding='utf-8') as f:
    for i in range(200):
        configs = "python run.py --config configs/NHR/xzq_%d.py --sample_voxels ../voxelfeature/data_xzq/xzq_%d.npz \n" % (i,i)
        f.write(configs)