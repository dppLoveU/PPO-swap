已完成只读检查，没有修改文件，也没有安装依赖。

**检查结果**

当前仓库要求的环境是：

```text
python=3.9.15
numpy=1.21.5
pytorch=1.13.0
pytorch-cuda=11.6
pytorch-lightning=1.7.7
pyg=2.2.0
networkx=2.8.4
gurobi=10.0.0
yaml=0.2.5
```

本机当前默认环境不适合直接跑：

```text
Python 3.13.12
conda 26.1.1
numpy 未安装
```

所以第一优先级是新建独立 Conda 环境，不要用当前默认 Python 跑。

**推荐安装命令**

最贴近原仓库、用于完整训练和评估的命令：

```bash
conda create -n ppo-swap python=3.9.15 numpy=1.21.5 networkx=2.8.4 pytorch=1.13.0 pytorch-cuda=11.6 pytorch-lightning=1.7.7 pyg=2.2.0 gurobi=10.0.0 pyyaml -c pytorch -c pyg -c gurobi -c nvidia -c conda-forge -c defaults
conda activate ppo-swap
```

如果只是先验证数据生成链路，最小 smoke test 环境可以更轻：

```bash
conda create -n ppo-swap-smoke python=3.9.15 numpy=1.21.5 networkx=2.8.4 pyyaml -c conda-forge
conda activate ppo-swap-smoke
```

但这个轻环境不能训练 PPO，因为缺 PyTorch、PyG、Lightning。

**先运行哪个脚本验证仓库通**

第一步建议先验证数据生成，不碰模型训练：

```bash
python gen_data.py
```

理由：`gen_data.py` 只依赖 `numpy` 和 `networkx`，是仓库最前置、最容易定位环境问题的入口。它会生成：

```text
./data/train_100_1000/
./data/test_100_10/
```

注意：默认会生成 1000 个训练图和 10 个测试图，不算特别轻。如果只想做最小 smoke test、且不修改代码，可以在环境激活后运行：

```bash
python -c "from gen_data import batch_gen; batch_gen('./data/smoke_20_1/', 20, 1)"
```

这个不会改核心算法，只调用已有函数生成 1 个 20 节点小图，用来确认数据生成链路是通的。

之后再按顺序跑：

```bash
python train.py
```

但训练前要注意 `config/train.yaml` 默认是：

```yaml
accelerator: gpu
devices: [1]
```

如果机器没有第二张 GPU，会失败。这属于环境/配置问题，不是算法问题。评估阶段还依赖 Gurobi license，而且 `eval_*.yaml` 里的 PPO checkpoint 目前是 `todo_version`，需要训练后再指向真实 `.ckpt`。



train:
A. 准备执行的具体步骤

1. 检查当前仓库状态和现有配置  
   只读查看 `environment.yml`、`config/train.yaml`、数据目录是否已存在、Conda 环境列表。

2. 优先基于仓库自带 `environment.yml` 创建 Conda 环境  
   计划环境名：`ppo-swap`。  
   第一选择命令是 `conda env create -n ppo-swap -f environment.yml`。

3. 验证环境是否创建成功  
   激活/调用该环境，检查 Python 版本和关键包导入：`torch`、`torch_geometric`、`pytorch_lightning`、`networkx`、`numpy`、`yaml`。

4. 做最小数据 smoke test  
   不直接跑默认 `python gen_data.py`，因为它会生成 1000 个训练图。  
   用已有 `gen_data.batch_gen()` 生成一个小数据集，例如：

   ```text
   ./data/smoke_20_1/
   ```

5. 新建 smoke 训练配置  
   不覆盖 `config/train.yaml`，新建类似：

   ```text
   config/train_smoke.yaml
   ```

   目标是极小训练：CPU 或当前可用 GPU、少量 epoch、少量 steps、指向 `./data/smoke_20_1/`。

6. 运行最小训练 smoke test  
   使用：

   ```bash
   python train.py -c config/train_smoke.yaml
   ```

   只验证数据集、环境、模型、Lightning 训练链路能走通。

7. 根据 smoke 结果决定是否建议完整训练  
   如果 smoke 失败，先判断是环境问题、GPU 配置问题、依赖版本问题、数据格式问题，提出最小修复，不重构核心算法代码。

B. 预计会新增或修改哪些文件

预计只新增，不修改原始核心算法：

```text
config/train_smoke.yaml
```

可能会由命令生成运行产物：

```text
data/smoke_20_1/
logs/<smoke_log_name>/
```

暂不修改：

```text
train.py
model.py
swap_env.py
dataset.py
config/train.yaml
environment.yml
```

C. 第一条准备执行的命令

```powershell
Get-Content -Raw environment.yml
```

用途：先确认仓库自带环境文件内容，作为创建 Conda 环境的依据。