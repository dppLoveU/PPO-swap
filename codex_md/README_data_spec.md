# Data Delivery Specification for PPO-swap

本文档基于当前仓库中 `dataset.py`、`gen_data.py`、`utils.py`、`methods/ppo_swap.py` 的真实读取逻辑，定义把苏州网约车原始 CSV 数据预处理成当前 PPO-swap 框架可直接读取格式时必须满足的数据交付规范。

本文档只描述“当前代码不修改时”的输入格式要求。

## 1. 数据集目录结构

当前 `GraphDataset(data_path, fac_range)` 会根据 `data_path` 最后一段下划线后的数字推断样本数量：

```python
self.city_num = int(data_path.rstrip("/").split("_")[-1])
```

因此数据集目录名必须以样本数量结尾，例如：

```text
data/suzhou_100_10/
data/suzhou_grid_20/
data/test_100_10/
```

如果目录名是 `data/suzhou_100_10/`，代码会认为共有 `10` 个样本，并依次读取：

```text
data/suzhou_100_10/0/graph.pkl
data/suzhou_100_10/0/distance_m.pkl
data/suzhou_100_10/1/graph.pkl
data/suzhou_100_10/1/distance_m.pkl
...
data/suzhou_100_10/9/graph.pkl
data/suzhou_100_10/9/distance_m.pkl
```

每个样本目录必须包含：

```text
<data_path>/<sample_id>/graph.pkl
<data_path>/<sample_id>/distance_m.pkl
```

其中 `sample_id` 必须从 `0` 连续编号到 `city_num - 1`。

FRP 评估或 `GraphImpDataset` 运行后，代码可能额外生成：

```text
<data_path>/init/<city_id>_<p>.pkl
```

这个 `init/` 目录不是交付时必须提供的输入。如果不存在，代码会自动随机生成初始设施点并保存。

## 2. graph.pkl 的精确结构要求

`graph.pkl` 必须是用 `pickle.dump()` 保存的 NetworkX 图对象。

推荐类型：

```python
networkx.Graph
```

当前 `gen_data.py` 生成的就是 `nx.Graph()`：

```python
G = nx.Graph()
```

当前读取逻辑：

```python
G = pickle.load(open(f"{data_path}/{i}/graph.pkl", "rb"))
```

随后在 `preprocess_graph(graph, distance_m)` 中读取：

```python
nx.get_node_attributes(graph, "pos")
nx.get_node_attributes(graph, "x")
nx.get_node_attributes(graph, "y")
nx.get_node_attributes(graph, "pop")
graph.edges(data="length")
```

因此 `graph.pkl` 至少必须满足：

- 是 NetworkX 图对象。
- 节点编号应为连续整数 `0, 1, ..., n-1`。
- 节点顺序必须与 `distance_m.pkl` 的行列顺序一致。
- 每个节点必须有需求字段 `pop`。
- 每个节点必须有坐标字段，二选一：
  - 推荐：`pos`
  - 备选：`x` 和 `y`
- 每条边必须有长度字段 `length`。

## 3. distance_m.pkl 的精确结构要求

`distance_m.pkl` 必须是用 `pickle.dump()` 保存的二维数值矩阵。

当前读取逻辑：

```python
distance_m_i = torch.Tensor(
    pickle.load(open(f"{data_path}/{i}/distance_m.pkl", "rb"))
)
```

因此 `distance_m.pkl` 可以是：

- `numpy.ndarray`
- Python list of lists
- 其他能被 `torch.Tensor(...)` 转成二维 tensor 的数值结构

必须满足：

```text
shape = (n, n)
```

其中 `n` 是 `graph.pkl` 的节点数量。

矩阵含义：

```text
distance_m[i][j] = 节点 i 到节点 j 的最短路径距离或业务定义距离
```

当前代码会在 `dataset.py` 中归一化：

```python
distance_m = distance_m / distance_m.max()
```

因此要求：

- 必须是数值型。
- 不能包含字符串。
- 不应包含 `NaN`。
- 不应包含 `inf`。
- `distance_m.max()` 必须大于 `0`。
- 行列索引必须对应图节点编号 `0..n-1`。
- 对角线通常应为 `0`。
- 如果使用无向路网，矩阵通常应对称；当前代码不强制检查对称性。

## 4. 节点字段要求

### 4.1 必须字段：`pop`

每个节点必须包含：

```python
G.nodes[i]["pop"]
```

读取逻辑：

```python
city_pop = torch.Tensor(list(nx.get_node_attributes(graph, "pop").values()))
```

用途：

- `utils.get_cost()` 用它计算人口/需求加权距离成本。
- `DensitySampling.sample()` 用它按密度采样初始设施点。
- `methods/ppo_swap.py` 用它构造节点特征。

要求：

- 所有节点都必须有 `pop`。
- `pop` 必须是数字。
- 推荐 `pop > 0`。
- 至少所有节点的 `pop` 总和必须大于 `0`，否则 `DensitySampling` 会失效。

对苏州网约车 OD 数据的建议：

```text
如果暂时不改算法，必须把 OD 需求聚合成每个节点的单点需求 pop。
例如可使用每个区域的出发量、到达量、或出发量+到达量作为 pop。
当前代码不读取 OD 矩阵。
```

### 4.2 坐标字段方案 A：`pos`

推荐每个节点包含：

```python
G.nodes[i]["pos"] = (x, y)
```

或：

```python
G.nodes[i]["pos"] = numpy.array([x, y])
```

读取逻辑：

```python
if len(nx.get_node_attributes(graph, "pos")) > 0:
    coordinates = torch.Tensor(
        np.array(list(nx.get_node_attributes(graph, "pos").values()))
    )
```

要求：

- `pos` 应为长度为 2 的数值序列。
- 对所有节点都应提供 `pos`。
- 不要只给部分节点提供 `pos`。当前代码只检查 `pos` 属性数量是否大于 0，只要有任意节点有 `pos`，就会走 `pos` 分支。

### 4.3 坐标字段方案 B：`x` 和 `y`

如果不提供 `pos`，则每个节点必须包含：

```python
G.nodes[i]["x"]
G.nodes[i]["y"]
```

读取逻辑：

```python
node_x = torch.Tensor(np.array(list(nx.get_node_attributes(graph, "x").values())))
node_y = torch.Tensor(np.array(list(nx.get_node_attributes(graph, "y").values())))
coordinates = torch.stack([node_x, node_y], 1)
```

要求：

- 所有节点都必须有 `x`。
- 所有节点都必须有 `y`。
- `x` 和 `y` 必须是数字。

### 4.4 坐标归一化要求

代码会执行：

```python
scale = max(torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0])
coordinates = (coordinates - torch.min(coordinates, 0)[0]) / scale
```

因此要求：

- 坐标不能全部相同。
- 至少 `x` 方向或 `y` 方向有非零跨度。
- 坐标可以是经纬度、投影坐标或已经归一化的坐标；代码都会重新归一化到近似 `[0, 1]`。

## 5. 边字段要求

每条边必须包含：

```python
G.edges[u, v]["length"]
```

读取逻辑：

```python
edges = list(graph.edges(data="length"))
edge_index = torch.LongTensor([(u, v) for u, v, _ in edges]).T
edge_attr = torch.Tensor([e[-1] for e in edges])
edge_attr = edge_attr.reshape(-1, 1) / edge_attr.max()
```

要求：

- 每条边都必须有 `length`。
- `length` 必须是数字。
- 推荐 `length > 0`。
- `edge_attr.max()` 必须大于 `0`。
- 边的两个端点必须是图中的合法节点编号。

注意：

- 当前代码只把 `graph.edges()` 返回的边方向写入 `edge_index`。
- 如果使用 `nx.Graph`，NetworkX 会以无向边形式存储，但 `edge_index` 里只会出现一次 `(u, v)`。
- 如果后续希望消息传递显式双向，可在数据预处理阶段自行加入双向边或改代码；但“当前不改代码”的交付格式建议沿用 `nx.Graph`。

## 6. 必须字段与可选字段

### 6.1 必须提供

每个样本目录必须提供：

```text
graph.pkl
distance_m.pkl
```

`graph.pkl` 中每个节点必须提供：

```text
pop
pos
```

或者：

```text
pop
x
y
```

`graph.pkl` 中每条边必须提供：

```text
length
```

`distance_m.pkl` 必须提供：

```text
n x n 数值距离矩阵
```

### 6.2 可选提供

节点可选字段：

```text
任意其他业务字段
```

边可选字段：

```text
任意其他业务字段
```

当前代码不会读取这些额外字段。

`init/` 初始设施目录可选：

```text
<data_path>/init/<city_id>_<p>.pkl
```

如果不提供，`GraphImpDataset` 会自动生成。

## 7. 当前代码真实使用链路

### 7.1 GraphDataset

`GraphDataset` 在初始化时读取全部样本：

```python
for i in range(self.city_num):
    G = pickle.load(open(f"{data_path}/{i}/graph.pkl", "rb"))
    distance_m_i = torch.Tensor(
        pickle.load(open(f"{data_path}/{i}/distance_m.pkl", "rb"))
    )
    coordinates, road_net_data, distance_m_i, city_pop = preprocess_graph(G, distance_m_i)
```

每个 `__getitem__` 返回：

```python
(
    city_id,
    city_pop,
    p,
    distance_m,
    coordinates,
    road_net_data,
)
```

其中：

- `city_pop`: 从节点 `pop` 得到。
- `p`: 从 `fac_range` 得到。
- `distance_m`: 从 `distance_m.pkl` 得到并归一化。
- `coordinates`: 从 `pos` 或 `x/y` 得到并归一化。
- `road_net_data.edge_index`: 从图边 `(u, v)` 得到。
- `road_net_data.edge_attr`: 从边 `length` 得到并归一化。

### 7.2 GraphImpDataset

`GraphImpDataset` 继承 `GraphDataset`，额外返回初始设施点：

```python
(
    city_id,
    city_pop,
    p,
    distance_m,
    coordinates,
    road_net_data,
    init_facility,
)
```

如果没有现成初始设施文件，会执行：

```python
init_facility = np.random.choice(
    len(self.city_pops[city_id]), size=p, replace=False
)
```

因此要求：

```text
p <= 节点数量 n
```

### 7.3 utils.get_cost

成本函数只使用：

```python
facility_list
distance_m
city_pop
```

逻辑：

```python
total_cost = torch.sum(
    (distance_m[facility_list] * city_pop.flatten())[
        torch.argmin(distance_m[facility_list], axis=0),
        torch.arange(distance_m.shape[1]),
    ]
)
```

含义：

```text
每个节点分配到最近设施点，成本为 距离 * 节点 pop，最后求和。
```

当前代码不读取完整 OD 矩阵。如果苏州网约车数据仍希望不改算法直接接入，必须先把 OD 需求聚合为节点级 `pop`。

### 7.4 methods/ppo_swap.py

PPO 评估阶段使用 `GraphDataset` 返回的数据：

```python
city_id, city_pop, p, distance_m, coordinates, road_net_data = batch[:6]
```

并在 `solve()` / `solve_reloc()` 中构造模型输入。关键输入特征来自：

- `coordinates`
- `city_pop`
- `distance_m`
- `road_net_data.edge_index`
- `road_net_data.edge_attr`

其中模型节点特征最终为 7 维：

```text
[x, y, normalized_pop, mask, normalized_node_cost, facility_pop_feature, facility_cost_feature]
```

这也是当前 `config/train.yaml` 中：

```yaml
fac_c_in: 7
```

的来源。

## 8. 交付前检查清单

交付苏州处理结果前，至少检查：

```text
1. data_path 目录名最后一段下划线后是样本数量。
2. 样本目录从 0 连续编号到 city_num - 1。
3. 每个样本目录都有 graph.pkl 和 distance_m.pkl。
4. graph.pkl 是 NetworkX 图对象。
5. 节点编号是 0..n-1。
6. 每个节点都有 pop。
7. 每个节点都有 pos，或者每个节点都有 x 和 y。
8. 每条边都有 length。
9. distance_m.pkl 是 n x n 数值矩阵。
10. distance_m 行列顺序与节点编号一致。
11. 图中 length 最大值大于 0。
12. distance_m 最大值大于 0。
13. 坐标至少有一个方向存在非零跨度。
14. 评估或训练配置中的最大 p 不超过 n。
```

## 9. 最小样例

下面示例生成一个包含 4 个节点、1 个样本的数据集：

```text
data/suzhou_smoke_1/0/graph.pkl
data/suzhou_smoke_1/0/distance_m.pkl
```

示例代码：

```python
import os
import pickle

import networkx as nx
import numpy as np

data_path = "./data/suzhou_smoke_1/0"
os.makedirs(data_path, exist_ok=True)

G = nx.Graph()

G.add_node(0, pos=(0.0, 0.0), pop=100.0)
G.add_node(1, pos=(1.0, 0.0), pop=80.0)
G.add_node(2, pos=(0.0, 1.0), pop=120.0)
G.add_node(3, pos=(1.0, 1.0), pop=60.0)

G.add_edge(0, 1, length=1.0)
G.add_edge(0, 2, length=1.0)
G.add_edge(1, 3, length=1.0)
G.add_edge(2, 3, length=1.0)
G.add_edge(1, 2, length=1.5)

distance_m = np.array(
    [
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 0.0, 1.5, 1.0],
        [1.0, 1.5, 0.0, 1.0],
        [2.0, 1.0, 1.0, 0.0],
    ],
    dtype=float,
)

pickle.dump(G, open(f"{data_path}/graph.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(distance_m, open(f"{data_path}/distance_m.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
```

读取验证：

```python
from dataset import GraphDataset

dataset = GraphDataset("./data/suzhou_smoke_1/", "range(1, 4)")
item = dataset[0]

city_id, city_pop, p, distance_m, coordinates, road_net_data = item

print(city_id)
print(city_pop.shape)
print(p)
print(distance_m.shape)
print(coordinates.shape)
print(road_net_data.edge_index.shape)
print(road_net_data.edge_attr.shape)
```

预期：

```text
city_id = 0
city_pop.shape = torch.Size([4])
p = 1
distance_m.shape = torch.Size([4, 4])
coordinates.shape = torch.Size([4, 2])
road_net_data.edge_index.shape = torch.Size([2, number_of_edges])
road_net_data.edge_attr.shape = torch.Size([number_of_edges, 1])
```
