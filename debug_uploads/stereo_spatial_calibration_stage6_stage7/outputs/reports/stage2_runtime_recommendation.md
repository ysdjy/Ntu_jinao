# Stage 2 Runtime Recommendation

- Current `./isaaclab.sh -p` Python in this run: `/home1/banghai/miniconda3/envs/env_isaaclab/bin/python`
- Current conda env: `env_isaaclab`
- Current torch import: `True`
- Current isaacsim import: `True`
- Current isaaclab import: `True`

## Diagnosis

The active Python should be checked against the JSON report before running native Isaac scripts.

## Recommended Runtime

- `env_isaaclab` exists: `True`
- `env_isaaclab` Python: `/home1/banghai/miniconda3/envs/env_isaaclab/bin/python`
- `env_isaaclab` torch available: `True`
- `env_isaaclab` CUDA available: `True`
- `env_isaaclab` GPU: `Quadro RTX 8000`
- `env_isaaclab` Isaac Sim/Lab available: `True`

Use:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py
```

Do not install torch into base just to fix this project. Use the existing `env_isaaclab` environment.

## isaaclab.sh

No code change to `isaaclab.sh` is required if the correct conda environment is activated before running it. Only consider modifying `isaaclab.sh` if you intentionally want to force a global Python path for all IsaacLab work.

## FoundationStereo

- Repo: `/home1/banghai/IsaacLab/third_party/FoundationStereo`
- Checkpoint: `None`
- Checkpoint ready: `False`

Expected checkpoint locations include:
- `/home1/banghai/IsaacLab/third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth`
- `/home1/banghai/IsaacLab/third_party/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth`

Fallback command while checkpoint is missing:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
```
