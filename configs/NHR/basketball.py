_base_ = '../default.py'

expname = 'basketball'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset_val/basketball_ps',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

