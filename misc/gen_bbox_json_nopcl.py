import numpy as np
import os
import json
import glob



root_path = '/data/new_disk2/wangla/Dataset/dataset_newmask/dance_newmask'
log_path = '/data/new_disk2/wangla/tmp/NCVV/logs/NHR/dance_test/bbox.json'

os.makedirs(os.path.dirname(log_path),exist_ok=True)
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump({"xyz_min": [-2.,-2.,-2.],
               "xyz_max":[2.,2.,2.]}, f, ensure_ascii=False, indent=4)
#print(min_xyzs.shape)