_base_ = '../default.py'

expname = 'xzq_white_frames'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset_newmask/xzq_white',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

