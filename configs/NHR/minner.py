_base_ = '../default.py'

expname = 'minner'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/NeuralHuman/Minner_wla',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

