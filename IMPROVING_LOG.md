# Improving log

### Custom CUDA implementation for efficiency
Some intermediate steps are reimplemented in cuda (`lib/cuda/`), which improves training speed by 
**1.8\~3.5**. Below show the results with dense grid under `256^3` voxels and `160^3` voxels. *Telsa V100*, *RTX 2080 Ti*, and *RTX 1080 Ti* are tested. The PSNRs of different versions on different machines have about 0.2 PSNR drift. The training speeds of the final version are improved 2--3 times from the original native pytorch implementation.

---

| **num_voxels=256^3**    | lego  |       | mic   |       | ship  |       |
|--------------|-------|-------|-------|-------|-------|-------|
| **GPU=V100** | psnr  | `mm:ss` | psnr  | `mm:ss` | psnr  | `mm:ss` |
| native pytorch<br>[b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 35.51 | `15:10`        | 34.39 | `14:11`        | 30.05 | `17:04` |
| cuda re-impl. Adam optimizer<br>[d3783f4](https://github.com/sunset1995/DirectVoxGO/tree/d3783f4) | 35.47 | `08:54` (1.7x) | 34.34 | `06:41` (2.1x) | 30.05 | `10:23` (1.6x) |
| cuda re-impl.  rendering<br>[3de7a6d](https://github.com/sunset1995/DirectVoxGO/tree/3de7a6d) | 35.63 | `06:31` (2.3x) | 34.48 | `04:31` (3.1x) | 30.30 | `08:20` (2.0x) |
| prevent atomic add in alpha2weight<br>[4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) |  35.61 | `05:35` (2.7x) | 34.51 | `04:00` (3.5x) | 30.29 | `07:20` (2.3x) |
| |
| **GPU=2080Ti** |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | - | OOM | 34.44 | `18:01` | - | OOM |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 35.61 | `07:19` | 34.49 | `04:30` (4.0x) | 30.29 | `09:53` |
| |
| **GPU=1080Ti** |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 35.76 | `37:22` | 34.47 | `31:18` | 30.09 | `45:28` |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 35.62 | `14:32` (2.6x) | 34.50 | `08:55` (3.5x) | 30.29 | `21:00` (2.2x) |

```python
# The model&training config for the results above
coarse_train = dict(N_iters=5000)
fine_train = dict(pg_scale=[1000,2000,3000,4000,5000,6000])
fine_model_and_render = dict(num_voxels=256**3)
```

---

| **num_voxels=160^3**    | lego  |       | mic   |       | ship  |       |
|--------------|-------|-------|-------|-------|-------|-------|
| **GPU=V100** | psnr  | `mm:ss` | psnr  | `mm:ss` | psnr  | `mm:ss` |
| native pytorch<br>[b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 34.65 | `08:29`        | 33.19 | `07:04`        | 29.08 | `10:38`        |
| cuda re-impl.  Adam optimizer<br>[d3783f4](https://github.com/sunset1995/DirectVoxGO/tree/d3783f4) | 34.66 | `06:01` (1.4x) | 33.14 | `04:38` (1.5x) | 29.04 | `08:06` (1.3x) |
| cuda re-impl.  rendering<br>[3de7a6d](https://github.com/sunset1995/DirectVoxGO/tree/3de7a6d) | 34.56 | `04:50` (1.8x) | 33.10 | `03:22` (2.1x) | 29.19 | `06:31` (1.6x) |
| prevent atomic add in alpha2weight<br>[4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 34.58 | `03:58` (2.1x) | 33.12 | `03:00` (2.4x) | 29.17 | `05:46` (1.8x) |
| |
| **GPU=2080Ti** |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 34.68 | `11:27` | 33.18 | `09:19` | 29.13 | `14:35` |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 34.59 | `04:59` (2.3x) | 33.15 | `03:04` (3.0x) | 29.19 | `07:32` (1.9x) |
| |
| **GPU=1080Ti** |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 34.66 | `22:01` | 33.19 | `17:14` | 29.10 | `29:57` |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 34.56 | `10:29` (2.1x) | 33.11 | `06:21` (2.7x) | 29.18 | `16:48` (x1.8) |

```python
# The model&training config for the results above
coarse_train = dict(N_iters=5000)
fine_train = dict(pg_scale=[1000,2000,3000,4000])
fine_model_and_render = dict(num_voxels=160**3)
```

---

| **GPU=2080Ti** | lego  |       | mic   |       | ship  |       |
|--------------|-------|-------|-------|-------|-------|-------|
| **num_voxels=256^3** | psnr  | `mm:ss` | psnr  | `mm:ss` | psnr  | `mm:ss` |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | - | OOM | 34.44 | `18:01` | - | OOM |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 35.61 | `07:19` | 34.49 | `04:30` (4.0x) | 30.29 | `09:53` |
| | 35.67 | `06:36` | 34.55 | `04:11` | 30.27 | `09:17` |
| | 35.93 | `07:37` | 34.76 | `04:49` | 30.48 | `09:55` |
| |
| **num_voxels=160^3** |
| native pytorch [b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 34.68 | `11:27` | 33.18 | `09:19` | 29.13 | `14:35` |
| cuda re-impl. [4f4ac99](https://github.com/sunset1995/DirectVoxGO/tree/4f4ac99) | 34.59 | `04:59` (2.3x) | 33.15 | `03:04` (3.0x) | 29.19 | `07:32` (1.9x) |
| | 34.60 | `04:37` (2.5x) | 33.19 | `02:57` (3.2x) | 29.01 | `07:08` |
| | 34.71 | `05:13` | 33.27 | `03:22` | 29.23 | `07:30` |


---

### Extend for forward-facing scene
The model for forward-facing scene is implemented in `lib/dmpigo.py`. Some main modifications include:
- Use NeRF's NDC warping
- Use Multiplane Image
- The initial probability stopping at each plane is `1/(# of planes)`
- Skip coarse stage training as it don't help in forward-facing scene
- Adopt total variation loss or the quality would degrade

All config files are in `configs/llff/`. The based config (`configs/llff/llff_default.py`) are something like:
```python
data = dict(
    dataset_type='llff',    # use llff dataloader
    ndc=True,               # use ndc coordinate (only for forward-facing; not support yet)
    width=1008,             # enforce image width
    height=756,             # enforce image height
)

coarse_train = dict(
    N_iters=0,              # we don't need the coarse stage training
)

fine_train = dict(
    N_iters=25000,
    N_rand=4096,            # it seem that larger batch don't help
    pg_scale=[2000,4000,6000,8000],
    ray_sampler='flatten',
    tv_before=1e9,          # enable total variation loss
    tv_dense_before=10000,  # dense version of total variation loss for the first 10k iterations
    weight_tv_density=1e-5,
    weight_tv_k0=1e-5,
)

fine_model_and_render = dict(
    num_voxels=256**3,
    mpi_depth=128,          # the number of planes in Multiplane Image (work when ndc=True)
    rgbnet_dim=9,           # it seem that more rgbnet_dim don't help
    rgbnet_width=64,        # it seem that larger rgbnet_width don't help
    world_bound_scale=1,    # we don't have to slightly enlarge the ndc
    fast_color_thres=1e-3,  # the initial probability stopping at each plane is 1/mpi_depth
                            # so the original 1e-4 would be too passive here
)
```

**Results** (on single 2080Ti GPU):
- PSNR is `25.71` in averaged, which is better than SRN's `22.84` and LLFF's `24.13` but is not as good as NeRF's `26.50`.
- Train time is `5.5 minutes` in averaged, which is much faster than NeRF's many hours.
- per-scene breakdown:
    | scene | psnr  | `mm:ss` |
    |-------|-------|---------|
    | fern     | 24.62 | `06:12` |
    | flower   | 27.33 | `05:28` |
    | fortress | 29.79 | `05:38` |
    | horns    | 26.73 | `05:23` |
    | leaves   | 20.75 | `04:36` |
    | orchids  | 19.96 | `05:26` |
    | room     | 30.61 | `05:55` |
    | trex     | 25.85 | `05:07` |
