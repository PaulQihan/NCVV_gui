_base_ = '../default.py'

expname = 'xzq'
basedir = './logs/NHR'
half_res = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset_newmask/xzq_dataset_newmask',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

