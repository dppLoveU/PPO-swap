# PPO-swap Smoke Reproduction Record

本文档记录当前仓库在不接入苏州数据、不修改核心算法代码的前提下，完成最小 smoke 复现的环境、命令、配置差异与产物。

## 1. 成功运行依赖的环境版本

当前成功 smoke 运行使用 Conda 环境：

```text
conda env: ppo-swap-smoke
python: 3.9.15
torch: 1.13.1+cpu
torch-geometric: 2.2.0
pytorch-lightning: 1.7.7
numpy: 1.21.5
networkx: 2.8.4
pyyaml: 6.0.2
scipy: 1.10.1
pip: 24.0
```

## 2. 与原论文环境的偏差

当前记录只是 smoke 复现，用于验证原始仓库的数据生成、训练入口、模型、环境和 checkpoint 保存链路能否跑通。

当前环境与原论文/原仓库默认环境存在以下偏差：

- 当前运行方式是 Windows Conda 环境 + WSL 仓库路径的混合调用。
- 原仓库 `environment.yml` 指定 `pytorch=1.13.0`、`pytorch-cuda=11.6`、`pyg=2.2.0`。
- 当前 Windows Conda 平台无法直接用原始 `environment.yml` 求解 `pyg=2.2.0`。
- Conda 版 `pytorch=1.13.0` 在当前机器上出现 `shm.dll` 加载失败。
- 因此 smoke 环境使用 `torch==1.13.1+cpu`，这是为了跑通链路做的最小兼容性修复。
- 当前 smoke 环境不代表完整论文实验环境，也不代表论文指标复现环境。
- 当前 smoke 使用 CPU，只验证代码链路，不评估训练性能和论文结果质量。
- 机器虽然有 RTX 5070 Ti，但本次记录对应的是已经成功跑通的 CPU smoke 结果；GPU 复现建议另建 CUDA 环境和独立 GPU smoke 配置。

## 3. 已成功执行过的命令

### 3.1 创建最小 smoke 环境

命令：

```powershell
conda create -n ppo-swap-smoke python=3.9.15 numpy=1.21.5 networkx=2.8.4 pyyaml -c conda-forge -y
```

说明：

```text
创建一个只包含基础数据生成依赖的 Python 3.9 smoke 环境。
```

### 3.2 验证基础依赖

命令：

```powershell
conda run -n ppo-swap-smoke python -c "import sys, numpy, networkx, yaml; print(sys.version); print('numpy', numpy.__version__); print('networkx', networkx.__version__); print('yaml ok')"
```

成功输出包含：

```text
python 3.9.15
numpy 1.21.5
networkx 2.8.4
yaml ok
```

### 3.3 生成最小 smoke 数据

命令：

```powershell
conda run -n ppo-swap-smoke python -c "import sys; sys.path.insert(0, r'\\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap'); from gen_data import gen_gabriel_graph; gen_gabriel_graph(r'\\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap\data\smoke_20_1\0', 123, 20); print('generated smoke_20_1')"
```

说明：

```text
调用仓库已有 gen_gabriel_graph()，生成 1 个 20 节点图。
未运行默认 gen_data.py，避免生成 1000 个训练图。
```

### 3.4 验证 smoke 数据

命令：

```powershell
conda run -n ppo-swap-smoke python -c "import pickle; p=r'\\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap\data\smoke_20_1\0'; G=pickle.load(open(p+'\\graph.pkl','rb')); D=pickle.load(open(p+'\\distance_m.pkl','rb')); print(G.number_of_nodes(), G.number_of_edges(), D.shape)"
```

成功输出：

```text
20 36 (20, 20)
```

### 3.5 安装训练 smoke 依赖

命令：

```powershell
conda run -n ppo-swap-smoke python -m pip install pip==24.0
conda run -n ppo-swap-smoke python -m pip install torch==1.13.1 pytorch-lightning==1.7.7 torchmetrics==0.10.3
conda run -n ppo-swap-smoke python -m pip install six scipy==1.10.1
```

说明：

```text
pip 固定为 24.0 是为了兼容 pytorch-lightning==1.7.7 的旧 metadata。
torch==1.13.1+cpu 是当前 Windows 环境下的最小兼容性修复。
```

### 3.6 安装 PyG

命令：

```powershell
conda run -n ppo-swap-smoke python -m pip install https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_scatter-2.1.1%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_sparse-0.6.17%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_cluster-1.6.1%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt113cpu-cp39-cp39-win_amd64.whl
conda run -n ppo-swap-smoke python -m pip install torch-geometric==2.2.0
```

说明：

```text
直接安装 PyG 官方预编译 wheel，避免在本机编译 torch_scatter、torch_sparse、torch_cluster。
```

### 3.7 验证训练依赖

命令：

```powershell
conda run -n ppo-swap-smoke python -c "import torch, torch_geometric, pytorch_lightning as pl; from torch_geometric.nn import GraphConv; from torch_geometric.data import Data, Batch; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('pl', pl.__version__); print(GraphConv)"
```

成功输出包含：

```text
torch 1.13.1+cpu
pyg 2.2.0
pl 1.7.7
<class 'torch_geometric.nn.conv.graph_conv.GraphConv'>
```

### 3.8 运行最小训练 smoke test

命令：

```powershell
cmd /c "pushd \\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap && conda run -n ppo-swap-smoke python -c ""from utils import get_config; from train import train_ppo; train_ppo(get_config(['-c','config/train_smoke.yaml']))"" && popd"
```

说明：

```text
train.py 的 __main__ 当前硬编码读取 config/train.yaml。
为了不修改核心算法代码，smoke 运行采用直接调用 train_ppo(get_config(...)) 的方式显式传入 config/train_smoke.yaml。
```

成功输出包含：

```text
Trainer.fit stopped: max_epochs=1 reached.
```

## 4. 新增文件和运行产物

### 4.1 新增配置文件

```text
config/train_smoke.yaml
```

### 4.2 新增 smoke 数据

```text
data/smoke_20_1/0/graph.pkl
data/smoke_20_1/0/distance_m.pkl
```

数据规模：

```text
nodes: 20
edges: 36
distance matrix: 20 x 20
```

### 4.3 新增训练日志与 checkpoint

```text
logs/smoke_test/config.yaml
logs/smoke_test/hparams.yaml
logs/smoke_test/events.out.tfevents.*
logs/smoke_test/checkpoints/epoch=0-step=11.ckpt
logs/smoke_test/checkpoints/last.ckpt
```

### 4.4 新增 PMP smoke 评估配置与结果

```text
config/eval_pmp_smoke.yaml
data/smoke_20_1/result_pmp_smoke/ppo_swap_1_2_2_smoke_test/0_5.pkl
data/smoke_20_1/result_pmp_smoke/ppo_swap_1_2_2_smoke_test/0_10.pkl
data/smoke_20_1/result_pmp_smoke/ppo_swap_1_2_2_smoke_test/0_15.pkl
```

## 5. train_smoke.yaml 相对原始 train.yaml 的关键差异

原始配置文件：

```text
config/train.yaml
```

smoke 配置文件：

```text
config/train_smoke.yaml
```

关键差异：

| 配置项 | 原始 train.yaml | smoke train_smoke.yaml | 目的 |
|---|---:|---:|---|
| `log_name` | 未固定，默认时间戳 | `smoke_test` | 固定 smoke 输出目录 |
| `ppo.batch_size` | `64` | `4` | 降低训练负担 |
| `ppo.steps_per_epoch` | `1024` | `44` | 超过环境默认 episode 长度 40，避免统计除零 |
| `ppo.nb_optim_iters` | `4` | `1` | 缩短 smoke 时间 |
| `ppo.model_params.c_hidden` | `128` | `16` | 缩小模型 |
| `ppo.model_params.c_out` | `128` | `16` | 缩小模型 |
| `ppo.model_params.num_layers` | `3` | `2` | 缩小模型 |
| `ppo.data_path` | `./data/train_100_1000/` | `./data/smoke_20_1/` | 使用最小数据集 |
| `ppo_trainer.max_epochs` | `300` | `1` | 只验证链路 |
| `ppo_trainer.accelerator` | `gpu` | `cpu` | 当前 smoke 环境使用 CPU |
| `ppo_trainer.devices` | `[1]` | `1` | CPU 单设备 |
| `ppo_trainer.auto_select_gpus` | `False` | 移除 | CPU smoke 不需要 |
| `ppo_trainer.enable_checkpointing` | 未显式设置 | `true` | 保留 checkpoint 产物 |

保持不变的核心 PPO 参数：

```text
gamma
lam
lr
lr_gamma
clip_ratio
clip_decay
ent_weight
critic_weight
gradient_clip_val
fac_c_in
layer_name
edge_dim
heads
```

## 6. 从零复现当前 smoke 结果

### Step 1: 创建 smoke Conda 环境

```powershell
conda create -n ppo-swap-smoke python=3.9.15 numpy=1.21.5 networkx=2.8.4 pyyaml -c conda-forge -y
```

### Step 2: 安装训练依赖

```powershell
conda run -n ppo-swap-smoke python -m pip install pip==24.0
conda run -n ppo-swap-smoke python -m pip install torch==1.13.1 pytorch-lightning==1.7.7 torchmetrics==0.10.3
conda run -n ppo-swap-smoke python -m pip install six scipy==1.10.1
```

### Step 3: 安装 PyG

```powershell
conda run -n ppo-swap-smoke python -m pip install https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_scatter-2.1.1%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_sparse-0.6.17%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_cluster-1.6.1%2Bpt113cpu-cp39-cp39-win_amd64.whl https://data.pyg.org/whl/torch-1.13.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt113cpu-cp39-cp39-win_amd64.whl
conda run -n ppo-swap-smoke python -m pip install torch-geometric==2.2.0
```

### Step 4: 生成 smoke 数据

```powershell
conda run -n ppo-swap-smoke python -c "import sys; sys.path.insert(0, r'\\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap'); from gen_data import gen_gabriel_graph; gen_gabriel_graph(r'\\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap\data\smoke_20_1\0', 123, 20); print('generated smoke_20_1')"
```

### Step 5: 确认 smoke 配置

确保存在：

```text
config/train_smoke.yaml
```

关键配置：

```yaml
log_name: smoke_test

ppo:
    data_path: ./data/smoke_20_1/
    batch_size: 4
    steps_per_epoch: 44

ppo_trainer:
    max_epochs: 1
    accelerator: cpu
    devices: 1
```

### Step 6: 运行训练 smoke test

```powershell
cmd /c "pushd \\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap && conda run -n ppo-swap-smoke python -c ""from utils import get_config; from train import train_ppo; train_ppo(get_config(['-c','config/train_smoke.yaml']))"" && popd"
```

### Step 7: 检查输出

成功后应看到：

```text
Trainer.fit stopped: max_epochs=1 reached.
```

并生成：

```text
logs/smoke_test/checkpoints/epoch=0-step=11.ckpt
logs/smoke_test/checkpoints/last.ckpt
```

## 7. PMP smoke 评估

### 7.1 评估配置

PMP smoke 评估使用独立配置文件：

```text
config/eval_pmp_smoke.yaml
```

关键配置：

```yaml
init_num: 1
iter_num: 2
swap_num: 2
device: cpu

data_path: ./data/smoke_20_1/
fac_range: "range(5, 16, 5)"
save_path: ./data/smoke_20_1/result_pmp_smoke/

methods:
  PPO-swap:
    run_fn: run_ppo_swap
    init_num: 1
    iter_num: 2
    swap_num: 2
    ckpt: ./logs/smoke_test/checkpoints/last.ckpt
    name: smoke_test
    device: cpu
```

### 7.2 为什么不直接运行 eval_pmp.py

本次 smoke 评估没有直接运行原始 `eval_pmp.py` 主入口，原因是：

- `eval_pmp.py` 会先运行 Gurobi baseline，可能被 Gurobi license 阻塞。
- `eval_pmp.py` 内部硬编码 `GraphDataset(data_path, "range(5, 41, 5)")`。
- 当前 smoke 数据只有 20 个节点，`p=25/30/35/40` 不合法。

因此本次采用最小替代方案：直接读取 `config/eval_pmp_smoke.yaml`，构造 `GraphDataset`，调用 `methods.ppo_swap.run_ppo_swap`。这样可以验证 checkpoint 加载、PPO 推理和结果保存链路，不修改核心算法代码。

### 7.3 运行 PMP smoke 评估

命令：

```powershell
cmd /c "pushd \\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap && conda run -n ppo-swap-smoke python -c ""from utils import get_config; from dataset import GraphDataset; from methods.ppo_swap import run_ppo_swap; cfg=get_config(['-c','config/eval_pmp_smoke.yaml']); ds=GraphDataset(cfg['data_path'], cfg['fac_range']); m=cfg['methods']['PPO-swap']; p=run_ppo_swap(dataset=ds, save_path=cfg['save_path'], **m); print('PMP smoke result path:', p)"" && popd"
```

成功输出包含：

```text
Running ppo_swap_1_2_2_smoke_test
PMP smoke result path: ./data/smoke_20_1/result_pmp_smoke//ppo_swap_1_2_2_smoke_test
```

### 7.4 PMP smoke 评估结果

实际生成目录：

```text
data/smoke_20_1/result_pmp_smoke/ppo_swap_1_2_2_smoke_test/
```

实际生成文件：

```text
0_5.pkl
0_10.pkl
0_15.pkl
```

这些文件分别对应 smoke 数据中的 `city_id=0` 和设施数量 `p=5/10/15`。

### 7.5 示例 pkl 结构

随机读取 `0_5.pkl` 可以正常反序列化。

读取命令：

```powershell
cmd /c "pushd \\wsl.localhost\Ubuntu-22.04\home\yuri\projects\PPO-swap && conda run -n ppo-swap-smoke python -c ""import pickle; p='data/smoke_20_1/result_pmp_smoke/ppo_swap_1_2_2_smoke_test/0_5.pkl'; obj=pickle.load(open(p,'rb')); print(type(obj)); print(obj.__dict__); print('facility_list_type', type(obj.facility_list)); print('facility_list', obj.facility_list); print('time', obj.time); print('cost', obj.cost)"" && popd"
```

示例输出：

```text
<class 'results.PMPSolution'>
{'time': 0.004001617431640625, 'facility_list': array([11,  4,  5, 18, 15]), 'cost': 326619.53125}
facility_list_type <class 'numpy.ndarray'>
facility_list [11  4  5 18 15]
time 0.004001617431640625
cost 326619.53125
```

基本结构：

```text
type: results.PMPSolution
fields:
  time: float
  facility_list: numpy.ndarray
  cost: float
```

### 7.6 PMP smoke 评估结论

```text
checkpoint loaded: yes
PPO inference completed: yes
result files generated: yes
blocking issue: none
```

## 8. 注意事项

- 本 smoke 复现不接入任何苏州数据。
- 本 smoke 复现未修改核心算法代码。
- 本 smoke 复现只验证链路可运行，不代表论文指标复现。
- 当前使用 CPU 环境，不适合用于完整训练性能评估。
- 完整训练前建议准备 Linux/WSL Conda 环境或 GPU 环境，再尽量回到原始 `environment.yml` 的 CUDA 依赖组合。
