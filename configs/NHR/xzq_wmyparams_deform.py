_base_ = '../default.py'

expname = 'xzq_wmyparams_deform_tv_from0_1e-3'
basedir = './logs/NHR'
half_res = True

train_mode = 'individual'
fix_rgbnet = True
use_deform="grid"
frame_num=200
data = dict(
    datadir='/data/new_disk2/wangla/Dataset/dataset_newmask/xzq_white',
    dataset_type='NHR',
    inverse_y=True,
    white_bkgd=True,
)

deform_from_start=True

#没有其他wmy 参数只是因为光学deformation不需要
fine_train=dict(
tv_every=1,# count total variation loss every tv_every step
tv_after=1000,# count total variation loss from tv_from step
tv_before=12000,         # count total variation before the given number of iterations
tv_dense_before=12000,      # count total variation densely before the given number of iterations
weight_tv_deform=1.,    # weight of total variation loss of density voxel grid
lrate_deformation_field=1e-3, #paper 是1e-3 但我们是两帧之间，他是所有帧，可能不一样？
lrate_decay=4,
pg_scale=[ 2000, 4000, 6000],
)

fine_model_and_render=dict(
 num_voxels=180**3,
  num_voxels_base=180**3,
  rgbnet_dim=12,
)