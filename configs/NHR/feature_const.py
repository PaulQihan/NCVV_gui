_base_ = '../default.py'

expname = 'feature_const_fromframe_seed202200804'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True

frame_num=1

data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset/dance_new',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)
