_base_ = '../default.py'

expname = 'lhm_fullreso'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/NeuralHuman/lhm',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

