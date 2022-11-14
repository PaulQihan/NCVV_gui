_base_ = '../default.py'

expname = 'eve'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

frame_num=600

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/NeuralHuman/eve_gangnam',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

