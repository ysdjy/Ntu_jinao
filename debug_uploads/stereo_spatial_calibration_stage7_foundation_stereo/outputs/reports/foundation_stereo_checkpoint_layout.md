# FoundationStereo Checkpoint Layout

- repo_path: `/home1/banghai/IsaacLab/third_party/FoundationStereo`
- repo_exists: `True`
- readme_path: `/home1/banghai/IsaacLab/third_party/FoundationStereo/readme.md`
- default_model_dir: `/home1/banghai/IsaacLab/third_party/FoundationStereo/pretrained_models/23-51-11`
- ckpt_path: `/home1/banghai/IsaacLab/third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth`
- cfg_path: `/home1/banghai/IsaacLab/third_party/FoundationStereo/pretrained_models/23-51-11/cfg.yaml`
- checkpoint_ready: `True`

## Confirmed From Local Repository

- The local repo uses lowercase `readme.md`, not root `README.md`.
- `readme.md` says to put the entire model folder, for example `23-51-11`, under `./pretrained_models/`.
- `scripts/run_demo.py` defaults `--ckpt_dir` to `../pretrained_models/23-51-11/model_best_bp2.pth`.
- `scripts/run_demo.py` loads `cfg.yaml` from the checkpoint parent directory.
- Inputs are left/right rectified, undistorted stereo RGB images; left/right must not be swapped.
- The demo converts disparity to depth with `depth = K[0,0] * baseline / disp`.

## Official Model Links Found In README

- `23-51-11`: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing
- `11-33-40`: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing

## Missing Files

- None
