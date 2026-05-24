# Franka State-Machine Cerebellum for VLM Skill Blueprint Execution

基于 Isaac Lab / Isaac Sim 4.5 的 Franka 状态机小脑模块，用于执行 VLM 或人工编写的技能蓝图 JSON，并为每个小技能采集执行性能标签（predictor 训练数据）。

本仓库**不包含** Isaac Lab 官方源码、仿真数据集或模型权重；需在已安装 Isaac Lab 的环境中，通过 `isaaclab.sh` 调用本模块脚本运行。

---

## 当前功能

| 阶段 | 说明 |
|------|------|
| **Stage I** | JSON skill plan → 状态机小脑 → pick/place → episode / trajectory 日志 |
| **Stage II** | skill blueprint JSON → sequence / condition 状态图 → primitive skills → `predictor_dataset.jsonl` |
| **Parallel** | 同一控制循环内并行收敛 `position_goal` 与 `orientation_goal`（位置 + 姿态） |

---

## 支持的小技能（Primitive Skills）

`move_above` · `reach` · `descend` · `grasp` · `lift` · `place` · `retreat` · `wait` · `align_orientation`

## 支持的逻辑类型

`sequence` · `condition` · `parallel`（第一版 `parallel_mode`: `all_success`）

---

## 环境要求

- **Isaac Sim 4.5**
- **Isaac Lab v2.0.x / v2.1.x**（兼容 IK-Rel Franka Lift 环境）
- Python 环境以 Isaac Lab 官方 Conda 环境为准（如 `isaaclab45`）
- 推荐任务：`Isaac-Lift-Cube-Franka-IK-Rel-v0`

---

## 目录结构

```text
source/standalone/franka_state_machine_cerebellum/
├── configs/
│   ├── example_skill_plan.json              # Stage I 示例
│   ├── example_skill_blueprint.json         # Stage II baseline（无 parallel）
│   └── example_skill_blueprint_parallel.json
├── datasets/.gitkeep                       # 本地输出占位（不提交数据）
├── run_state_machine_cerebellum.py          # Stage I 入口
├── run_skill_blueprint_executor.py          # Stage II 入口
└── …                                        # 状态机、蓝图 loader、primitive skills 等
```

---

## 运行示例

在 **Isaac Lab 根目录**下执行（将路径替换为你的 Isaac Lab 安装位置）：

```bash
cd /path/to/IsaacLab
conda activate isaaclab45

# Stage I：JSON skill plan → pick/place
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_state_machine_cerebellum.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 5 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --output_dir ./datasets/state_machine_stage1 \
  --save_trajectory true \
  --target_mode custom_tabletop \
  --seed 0

# Stage II：skill blueprint（baseline，无 parallel）
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 5 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint.json \
  --output_dir ./datasets/state_machine_stage2_blueprint \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true

# Stage II + Parallel：位置与姿态并行收敛
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 5 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json \
  --output_dir ./datasets/state_machine_stage2_parallel \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

若本仓库克隆到独立目录，请将上述脚本路径改为你本地的 `franka_state_machine_cerebellum` 绝对路径，或在 Isaac Lab 中创建符号链接指向本仓库模块。

---

## 数据输出（默认不提交 GitHub）

运行后在 `--output_dir` 指定目录生成：

| 文件 / 目录 | 说明 |
|-------------|------|
| `episodes.jsonl` | 每个 episode 一条记录 |
| `predictor_dataset.jsonl` | Stage II 每个 skill/parallel 节点一条 predictor 样本 |
| `summary.json` | 汇总统计 |
| `trajectories/` | 逐步轨迹（可选） |

以上均由 `.gitignore` 排除，请勿将大规模仿真输出推送到 GitHub。

---

## 注意事项

1. 本模块通过 **standalone 脚本** 调用 Isaac Lab 环境，不修改 Isaac Lab 官方核心代码。
2. 不包含 RL / BC / VLM / predictor **训练**代码，仅做执行与数据采集。
3. 低显存环境建议加 `--headless` 与 `--kit_args="--/rtx/verifyDriverVersion/enabled=false"`。
4. 示例 JSON 蓝图位于 `configs/`，可直接修改或通过 watch 模式热加载（Stage II）。

---

## License

Research / academic use. Isaac Lab and Isaac Sim are subject to their respective NVIDIA licenses.
