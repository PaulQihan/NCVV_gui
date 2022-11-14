import numpy as np
import os
import json


root_path = '../configs/NHR'

for i in range(200):
    configs = "_base_ = '../default.py' \nexpname = 'xzq_%d'\nbasedir = '/scratch/leuven/346/vsc34668/xzq_dataset_newmask/logs/NHR'\nhalf_res = True\ndata = dict(\n        datadir='/scratch/leuven/346/vsc34668/xzq_dataset_newmask',\n        frame = %d,\n        dataset_type='NHR',\n        inverse_y=True,\n        white_bkgd=True,\n    )\n" % (i,i)
    with open(os.path.join(root_path,'xzq_%d.py' % i), 'w', encoding='utf-8') as f:
            f.write(configs)



with open('../run.sh','w') as f:
    for i in range(200):
        f.write('python run.py --config configs/NHR/xzq_%d.py --render_test\n' % i)


