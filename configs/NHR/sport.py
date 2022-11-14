_base_ = '../default.py'

expname = 'sport'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset/sport_1_mask',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

