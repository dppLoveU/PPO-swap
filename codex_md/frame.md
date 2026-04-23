我没有修改任何文件，也没有安装依赖；下面是只读分析结果。

**1. 仓库整体目标**

这个仓库实现论文 *“Unified and Generalizable Reinforcement Learning for Facility Location Problems on Graphs”* 里的 `PPO-swap` 方法。核心目标是在图结构城市网络上，用强化学习通过“交换设施点”的方式求解两类设施选址问题：

- `PMP`：p-median problem，从候选节点中选 `p` 个设施点，使人口加权到最近设施的距离成本最小。
- `FRP`：facility relocation problem，给定已有设施点，在有限迁移次数内重新布点，降低总成本。

当前数据默认是合成 Gabriel 图，每个节点有坐标和人口 `pop`，每条边有长度 `length`，目标函数本质是：

```text
sum_over_nodes(pop[node] * distance(node, nearest_facility))
```

**2. 文件分工**

训练：

- `train.py`：训练入口，读取 `config/train.yaml`，启动 PyTorch Lightning `Trainer`
- `model.py`：PPO LightningModule、Actor-Critic、GNN、轨迹采样和 PPO loss
- `swap_env.py`：训练环境，负责 reset/step/reward

评估：

- `eval_pmp.py`：PMP 评估入口
- `eval_frp.py`：FRP 评估入口
- `results.py`：保存解、计算 gap/improvement、导出 CSV
- `methods/*.py`：各类 baseline 和 PPO-swap solver，包括 Gurobi、greedy、random、SA、Maranzana、BestResponse、FR2FP 等

数据生成与读取：

- `gen_data.py`：生成合成图数据，输出 `graph.pkl` 和 `distance_m.pkl`
- `dataset.py`：读取图数据，预处理坐标、人口、边、距离矩阵
- `utils.py`：成本函数 `get_cost`、配置读取、设施初始化采样

模型：

- `model.py`：`GraphFeatureExtractor`、`ActorCritic`、`PPOLightning`
- `methods/ppo_swap.py`：评估阶段加载训练好的 PPO checkpoint 并执行 swap

环境：

- `swap_env.py`：`SwapEnv`，训练时的 RL 环境

配置：

- `config/train.yaml`：训练超参、模型参数、训练数据路径、GPU 配置
- `config/eval_pmp.yaml`：PMP 评估配置
- `config/eval_frp.yaml`：FRP 评估配置
- `environment.yml`：Conda 依赖
- `.codex/config.toml`：看起来是 Codex/IDE 相关配置，不是算法运行配置

**3. 最小复现路径**

最小路径按仓库 README 是：

```bash
python gen_data.py
python train.py
python eval_pmp.py
python eval_frp.py
```

但实际有几个关键前提：

1. 依赖来自 `environment.yml`  
   主要包括 Python 3.9、PyTorch 1.13、PyTorch Geometric、PyTorch Lightning、NetworkX、Gurobi、YAML。

2. 先生成数据  
   `gen_data.py` 默认生成：

```text
./data/train_100_1000/
./data/test_100_10/
```

每个图目录类似：

```text
data/train_100_1000/0/graph.pkl
data/train_100_1000/0/distance_m.pkl
```

3. 训练依赖 `config/train.yaml`  
   默认训练数据：

```yaml
data_path: ./data/train_100_1000/
```

默认 GPU 配置是：

```yaml
accelerator: gpu
devices: [1]
```

如果机器没有 GPU 1，这里会直接出问题。

4. 训练产物在：

```text
./logs/<log_name>/checkpoints/
```

5. 评估配置里的 PPO checkpoint 目前还是占位符：

```yaml
ckpt: todo_version
name: todo_version
```

所以直接跑 `eval_pmp.py` / `eval_frp.py` 时，PPO-swap 部分不会完整复现，必须先把 `ckpt` 指到训练得到的 `.ckpt` 文件。

6. 评估默认会跑 Gurobi  
   `eval_pmp.py` 和 `eval_frp.py` 都会先跑 Gurobi optimal，依赖 `gurobipy` 和可用 license。对 100 节点测试集应该还算可控，但这是复现评估的硬依赖。

**4. 接入苏州网约车 OD 数据最可能要改的文件**

最可能涉及这些文件，按优先级：

1. `gen_data.py`  
   把合成 Gabriel 图生成逻辑替换/扩展为苏州真实路网、网格/区域节点、OD 需求、距离矩阵生成逻辑。

2. `dataset.py`  
   当前只读取 `graph.pkl` 的节点 `pop` 和边 `length`。如果 OD 数据不是简单节点人口，而是 OD 矩阵或时段需求，需要改数据字段、预处理和返回值。

3. `utils.py`  
   `get_cost()` 当前是节点人口加权最近设施距离。如果苏州 OD 要表达“出行需求到服务点/枢纽/换乘点”的成本，目标函数大概率要改这里。

4. `swap_env.py`  
   训练 reward、状态特征、设施点成本分摊都写死为 `city_pop + distance_m + nearest facility`。OD 目标一变，这里必须同步。

5. `methods/ppo_swap.py`  
   评估阶段复制了一份和环境类似的 `_get_fac_data()` 与成本计算逻辑，接 OD 后这里也要同步，否则训练和评估口径会不一致。

6. `model.py`  
   如果节点特征从当前 7 维变成 OD/时段/区域类型等更多特征，需要改 `fac_c_in` 对应输入维度，可能也要调整网络结构。

7. `config/train.yaml`、`config/eval_pmp.yaml`、`config/eval_frp.yaml`  
   改 `data_path`、`fac_range`/设施数量范围、GPU、checkpoint、模型输入维度等。

8. `methods/gurobi.py` 以及其他 baseline  
   如果目标从 p-median 人口加权距离变成真正 OD 流量成本，Gurobi 和 greedy/random/BR/SA 等 baseline 的成本计算也要一起改，保证公平比较。

一个特别容易踩坑的点：`GraphDataset` 会从 `data_path` 最后的下划线后面解析图数量，例如 `train_100_1000` 会认为有 1000 个图。所以苏州数据目录命名或 `dataset.py` 的解析逻辑也要注意。

**5. 三阶段计划**

**阶段一：先复现**

- 创建 `environment.yml` 对应环境，确认 PyTorch、PyG、Gurobi 可用。
- 跑 `python gen_data.py`，确认生成 `train_100_1000` 和 `test_100_10`。
- 跑 `python train.py`，必要时先把 `config/train.yaml` 的 GPU 设备调成当前机器可用设备。
- 找到 `logs/.../checkpoints/*.ckpt`。
- 将评估配置里的 `ckpt` 和 `name` 指向训练产物，再跑 `eval_pmp.py` 和/或 `eval_frp.py`。
- 检查 `data/test_100_10/result_pmp/`、`results_frp_*/` 下的 CSV 和 pkl 结果。

**阶段二：再替换数据**

- 先决定苏州 OD 的建模口径：  
  是把 OD 聚合成节点 `pop`，保持原 p-median 目标；还是保留完整 OD 矩阵，改成流量加权目标。
- 写真实数据转换流程，产出仓库能读的基本结构：`graph.pkl`、`distance_m.pkl`。
- 保证图节点有坐标字段和需求字段，边有 `length`。
- 先用少量苏州样本构造一个小数据集，例如 `suzhou_100_1` 或 `suzhou_grid_1`。
- 只跑 `dataset.py` 读取链路和 `utils.get_cost()` 成本链路，确认维度、节点编号、距离矩阵都一致。
- 再跑 baseline 中最简单的 greedy/random，确认数据可用于求解。

**阶段三：再改环境**

- 如果仍然使用节点人口加权 p-median，主要改数据生成和配置，环境可少改。
- 如果使用完整 OD 目标，则统一修改 `utils.get_cost()`、`swap_env.py`、`methods/ppo_swap.py`、`methods/gurobi.py` 和其他 baseline。
- 扩展节点/边/OD 特征后，同步修改 `model.py` 输入维度和 `config/train.yaml` 的 `fac_c_in`。
- 用一个极小苏州样本做 smoke test：1 个图、少量节点、少量设施数。
- 确认训练 reward、评估 cost、Gurobi/baseline cost 三者口径一致。
- 最后再扩大到完整苏州 OD 数据，做训练和评估对比。