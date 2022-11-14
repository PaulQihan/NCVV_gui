_base_ = '../default.py'

expname = 'dance_new_wmy'
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

fine_train=dict(
tv_every=1,# count total variation loss every tv_every step
tv_after=1000,# count total variation loss from tv_from step
tv_before=12000,         # count total variation before the given number of iterations
tv_dense_before=12000,      # count total variation densely before the given number of iterations
weight_tv_density=0.000016,    # weight of total variation loss of density voxel grid
weight_tv_k0=0.0,       # weight of total variation loss of color/feature voxel grid
pg_scale=[ 2000, 4000, 6000],
)

fine_model_and_render=dict(
 num_voxels=180**3,
  num_voxels_base=180**3,
  rgbnet_dim=16,
)