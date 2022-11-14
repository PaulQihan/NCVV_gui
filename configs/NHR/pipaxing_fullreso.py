_base_ = '../default.py'

expname = 'pipaxing_fullreso'
basedir = './logs/NHR'
half_res = False

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset_newmask/pipaxing',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

