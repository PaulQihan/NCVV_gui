# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda-11.7 \
#     python test_decode_gui.py --config ./logs/NHR/xzq_wmyparams_deform_tv_res_cube_l1/config.py \
#         --model_path ./logs/NHR/xzq_wmyparams_deform_tv_res_cube_l1/dynamic+99 --render_360 100 \
#         --gui 

# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python debug_visualization.py --config configs/NHR/jywq.py --export_bbox_and_cams_only jywq_cam.npz

# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python tools/vis_train.py jywq_cam.npz /home/vrlab/workspace/NCVV/NCVV/logs/NHR/jywq_2_250/bbox.json

# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python run.py --config configs/NHR/jywq.py --export_coarse_only jywq_coarse.npz

# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python tools/vis_volume.py jywq_coarse.npz /home/vrlab/workspace/NCVV/NCVV/logs/NHR/jywq_2_250/bbox.json \
#                 0.001 --cam jywq_cam.npz



# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python debug_visualization.py --config /home/vrlab/workspace/NCVV/NCVV/logs/NHR/luoxi_1000_250/config.py --export_bbox_and_cams_only luoxi_cam.npz

# LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
#         python tools/vis_train.py luoxi_cam.npz /home/vrlab/workspace/NCVV/NCVV/logs/NHR/luoxi_1000_250/bbox.json

LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
        python run.py --config /home/vrlab/workspace/NCVV/NCVV/logs/NHR/luoxi_1000_250/config.py --export_coarse_only luoxi_coarse.npz

LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH CUDA_HOME=/usr/local/cuda \
        python tools/vis_volume.py luoxi_coarse.npz /home/vrlab/workspace/NCVV/NCVV/logs/NHR/luoxi_1000_250/bbox.json \
                0.001 --cam luoxi_cam.npz