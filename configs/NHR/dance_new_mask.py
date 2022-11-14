_base_ = '../default.py'

expname = 'dance_new_mask1024'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

frame_num=600

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset/dance_new',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

