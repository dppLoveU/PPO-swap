
# 苏州网约车数据处理流程说明

本文档目标是把苏州网约车原始订单 CSV 处理成两层输出：

- A. 中间层：清洗后的订单、区域映射、OD 矩阵
- B. 最终层：当前 PPO-swap 仓库可直接读取的 `graph.pkl` 和 `distance_m.pkl`

当前 PPO-swap 代码不直接读取完整 OD 矩阵。最终接入时，必须把 OD 信息聚合成节点级需求 `pop`，并保存为 NetworkX 图中的节点属性。

## 1. 原始字段

原始订单数据至少包含：

```text
fDepLongitude
fDepLatitude
fDestLongitude
fDestLatitude
fDepTime
fDestTime
fDriveMile
fDriveTime
AreaName
date
```

字段含义建议统一为：

| 字段 | 含义 |
|---|---|
| `fDepLongitude` | 上车点经度 |
| `fDepLatitude` | 上车点纬度 |
| `fDestLongitude` | 下车点经度 |
| `fDestLatitude` | 下车点纬度 |
| `fDepTime` | 上车时间 |
| `fDestTime` | 下车时间 |
| `fDriveMile` | 行驶里程 |
| `fDriveTime` | 行驶时长 |
| `AreaName` | 原始区域名或行政区字段 |
| `date` | 订单日期 |

## 2. 原始数据清洗步骤

### 2.1 字段标准化

统一字段名、类型和时间格式。

要求：

```text
经纬度字段转 float
fDriveMile 转 float
fDriveTime 转 float
fDepTime/fDestTime/date 转 datetime 或标准字符串
```

建议新增字段：

```text
dep_datetime
dest_datetime
dep_hour
dep_date
```

### 2.2 删除关键字段缺失订单

删除以下字段缺失的订单：

```text
fDepLongitude
fDepLatitude
fDestLongitude
fDestLatitude
fDepTime
fDestTime
```

如果要使用里程或时长构造距离，也需要删除：

```text
fDriveMile
fDriveTime
```

### 2.3 经纬度合法性过滤

至少过滤：

```text
经度为空
纬度为空
经度超出合理范围
纬度超出合理范围
```

苏州附近可使用一个粗略 bounding box，例如：

```text
longitude: 119.8 ~ 121.0
latitude: 30.7 ~ 32.2
```

实际范围由数据组根据业务边界确认。

### 2.4 时间合法性过滤

删除：

```text
fDestTime <= fDepTime
fDriveTime <= 0
fDriveMile < 0
```

建议过滤异常订单：

```text
行程时间过长
行程距离过长
平均速度异常
```

例如：

```text
duration_minutes = (dest_datetime - dep_datetime).minutes
speed_kmh = fDriveMile / duration_hours
```

可先使用宽松阈值：

```text
duration_minutes: 1 ~ 240
speed_kmh: 1 ~ 120
```

具体阈值应结合苏州网约车数据分布再调整。

### 2.5 去重

如果原始数据有订单 ID，应按订单 ID 去重。

如果没有订单 ID，可用以下字段组合做近似去重：

```text
fDepLongitude
fDepLatitude
fDestLongitude
fDestLatitude
fDepTime
fDestTime
fDriveMile
fDriveTime
```

### 2.6 输出清洗订单表

中间层清洗订单建议输出：

```text
intermediate/clean_orders.csv
```

建议字段：

```text
order_id
dep_lng
dep_lat
dest_lng
dest_lat
dep_datetime
dest_datetime
dep_date
dep_hour
drive_mile
drive_time
area_name
```

如果没有 `order_id`，可自行生成连续编号。

## 3. 空间离散化建议

目标是把连续经纬度订单点映射到有限个图节点。每个图节点代表一个区域、网格或交通分析区。

### 3.1 推荐方案 A：规则网格

适合快速接入 PPO-swap。

步骤：

```text
1. 确定苏州研究范围 bounding box
2. 设定网格边长，例如 500m、1km 或 2km
3. 将每个上车点和下车点映射到网格 ID
4. 每个有效网格作为一个节点
```

优点：

```text
实现简单
节点编号容易连续化
适合从订单点直接构图
```

缺点：

```text
不一定贴合行政区或真实路网
边界区域可能稀疏
```

### 3.2 推荐方案 B：行政区/交通小区

如果已有苏州行政区、街道、交通分析区或网约车运营区域 polygon，优先使用。

步骤：

```text
1. 准备区域 polygon
2. 对上车点、下车点做 point-in-polygon
3. 将订单映射到 origin_area_id 和 dest_area_id
4. 每个区域作为一个节点
```

优点：

```text
业务解释性强
区域需求更稳定
```

缺点：

```text
需要可靠边界数据
区域面积差异可能较大
```

### 3.3 节点编号要求

最终 PPO-swap 数据要求节点编号必须是：

```text
0, 1, 2, ..., n-1
```

因此无论使用网格还是区域，都必须输出区域映射表：

```text
intermediate/area_mapping.csv
```

建议字段：

```text
node_id
area_id
area_name
center_lng
center_lat
geometry_wkt
```

其中：

```text
node_id: 连续整数，从 0 开始
area_id: 原始区域 ID 或网格 ID
center_lng/center_lat: 节点中心点坐标
geometry_wkt: 可选，区域 polygon 或 grid polygon
```

## 4. 时间切片建议

PPO-swap 当前每个样本目录代表一个独立图样本。可以把不同日期或不同时段构造成多个样本。

### 4.1 最简单方案：全量订单聚合为一个样本

输出：

```text
data/suzhou_1/0/graph.pkl
data/suzhou_1/0/distance_m.pkl
```

适合第一版接入验证。

### 4.2 按日期切片

每一天一个样本：

```text
data/suzhou_daily_30/0/
data/suzhou_daily_30/1/
...
data/suzhou_daily_30/29/
```

注意：

```text
目录名最后一个下划线后的数字必须等于样本数量。
例如 suzhou_daily_30 表示 30 个样本。
```

### 4.3 按时段切片

可按早高峰、平峰、晚高峰、夜间切片：

```text
morning_peak: 07:00-10:00
daytime: 10:00-16:00
evening_peak: 16:00-20:00
night: 20:00-24:00
```

每个日期 + 时段可作为一个样本。

### 4.4 初次接入建议

建议先做：

```text
一个全量样本
一个工作日早高峰样本
一个工作日晚高峰样本
```

先验证框架能读取，再扩展到更多时间切片。

## 5. 从原始订单生成 OD 矩阵

### 5.1 为订单匹配 O 和 D 节点

对每条清洗后的订单：

```text
origin_node = 上车点所在区域或网格 node_id
dest_node = 下车点所在区域或网格 node_id
```

删除无法匹配到节点的订单。

清洗订单表中建议新增：

```text
origin_node
dest_node
```

### 5.2 生成 OD 计数矩阵

对于 `n` 个节点，生成：

```text
OD_count: shape = (n, n)
```

定义：

```text
OD_count[i, j] = 从节点 i 到节点 j 的订单数量
```

### 5.3 生成 OD 里程或时长矩阵

可选输出：

```text
OD_mile_sum[i, j] = 从 i 到 j 的总行驶里程
OD_time_sum[i, j] = 从 i 到 j 的总行驶时长
OD_mile_mean[i, j] = 从 i 到 j 的平均行驶里程
OD_time_mean[i, j] = 从 i 到 j 的平均行驶时长
```

建议中间层保存：

```text
intermediate/od_count.npy
intermediate/od_mile_mean.npy
intermediate/od_time_mean.npy
intermediate/od_pairs.csv
```

`od_pairs.csv` 建议字段：

```text
origin_node
dest_node
order_count
total_drive_mile
mean_drive_mile
total_drive_time
mean_drive_time
```

## 6. 从 OD 矩阵聚合节点需求 pop

当前 PPO-swap 不读取完整 OD 矩阵，只读取节点级 `pop`。

节点 `pop` 可以从 OD 聚合得到。推荐提供多个版本供实验选择，但最终 `graph.pkl` 中只能写入一个 `pop` 字段。

### 6.1 出发需求

```text
pop[i] = sum_j OD_count[i, j]
```

含义：

```text
节点 i 的出发订单量
```

适合设施服务出发需求。

### 6.2 到达需求

```text
pop[i] = sum_j OD_count[j, i]
```

含义：

```text
节点 i 的到达订单量
```

适合设施服务到达需求。

### 6.3 出发 + 到达需求

```text
pop[i] = sum_j OD_count[i, j] + sum_j OD_count[j, i]
```

含义：

```text
节点 i 的总活跃需求
```

推荐作为第一版默认方案。

### 6.4 加权需求

如果希望综合订单量、里程和时长，可定义：

```text
pop[i] = alpha * dep_count[i] + beta * dest_count[i] + gamma * total_drive_time_related[i]
```

第一版不建议过复杂。优先用：

```text
pop = dep_count + dest_count
```

### 6.5 零需求节点处理

当前代码中：

```python
DensitySampling.sample(city_pop, p)
```

会用 `pop` 作为采样概率来源。因此：

```text
pop 总和必须大于 0
```

建议：

```text
删除长期零需求节点
或给保留节点设置一个很小的 epsilon，例如 1e-6
```

为了业务解释性，第一版建议删除无订单覆盖的孤立零需求节点，并重新编号。

## 7. 构建 graph.pkl

`graph.pkl` 必须是 NetworkX 图对象。

### 7.1 节点

每个节点必须写入：

```python
G.add_node(node_id, pos=(center_lng, center_lat), pop=pop_value)
```

或：

```python
G.add_node(node_id, x=center_lng, y=center_lat, pop=pop_value)
```

推荐使用 `pos`。

字段要求：

```text
node_id: 连续整数 0..n-1
pos: 长度为 2 的数值坐标
pop: 数值需求
```

### 7.2 边

边表示区域之间的邻接或可达关系。

每条边必须写入：

```python
G.add_edge(u, v, length=length_value)
```

`length` 可以来自：

```text
区域中心点之间的球面距离
区域中心点之间的路网最短距离
OD 平均行驶距离
相邻网格距离
```

第一版建议：

```text
规则网格：相邻网格连边，length 使用中心点距离
区域 polygon：相邻区域连边，length 使用中心点距离或路网距离
```

要求：

```text
length > 0
```

### 7.3 图连通性

`distance_m.pkl` 通常来自图上的最短路径距离。建议保证图是连通的。

如果图不连通：

```text
networkx.shortest_path_length 可能无法得到所有 i,j 距离
distance_m 可能出现 inf
当前 PPO-swap 不接受 inf
```

第一版建议：

```text
只保留最大连通分量
或补充边使图连通
```

## 8. 构建 distance_m.pkl

`distance_m.pkl` 是 `n x n` 数值矩阵。

### 8.1 推荐做法：基于 graph 的 length 求最短路径

```python
p = dict(nx.shortest_path_length(G, weight="length"))
distance_m = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distance_m[i, j] = p[i][j]
```

### 8.2 可选做法：使用路网距离矩阵

如果已有苏州路网节点或区域中心点间路网距离，可直接构造：

```text
distance_m[i, j] = 区域 i 到区域 j 的路网最短距离
```

但必须确保：

```text
shape = (n, n)
行列顺序与 node_id 一致
distance_m.max() > 0
无 NaN
无 inf
```

### 8.3 不建议直接使用 OD 平均距离作为唯一 distance_m

OD 平均距离只覆盖有订单的 OD 对，稀疏且可能不对称。当前模型更适合稠密的节点间距离矩阵。

如果要使用 OD 平均距离，必须处理：

```text
没有订单的 i,j 对
异常值
非对称问题
```

第一版建议仍使用图上最短路径或路网最短路径。

## 9. 推荐输出目录结构

建议数据处理结果分为中间层和最终层。

```text
processed_suzhou/
  intermediate/
    clean_orders.csv
    area_mapping.csv
    od_count.npy
    od_mile_mean.npy
    od_time_mean.npy
    od_pairs.csv
    README_processing_notes.md

  ppo_swap/
    suzhou_smoke_1/
      0/
        graph.pkl
        distance_m.pkl

    suzhou_daily_30/
      0/
        graph.pkl
        distance_m.pkl
      1/
        graph.pkl
        distance_m.pkl
      ...
      29/
        graph.pkl
        distance_m.pkl
```

如果要直接放入当前仓库，可使用：

```text
data/suzhou_smoke_1/0/graph.pkl
data/suzhou_smoke_1/0/distance_m.pkl
```

或：

```text
data/suzhou_daily_30/0/graph.pkl
data/suzhou_daily_30/0/distance_m.pkl
...
data/suzhou_daily_30/29/graph.pkl
data/suzhou_daily_30/29/distance_m.pkl
```

注意：

```text
目录名最后的数字必须等于样本数量。
```

## 10. 给组员的任务说明

请按以下步骤处理苏州网约车原始 CSV：

### 任务 1：清洗订单

输入：

```text
原始网约车订单 CSV
```

至少使用字段：

```text
fDepLongitude
fDepLatitude
fDestLongitude
fDestLatitude
fDepTime
fDestTime
fDriveMile
fDriveTime
AreaName
date
```

输出：

```text
processed_suzhou/intermediate/clean_orders.csv
```

要求：

```text
1. 删除关键字段缺失订单。
2. 过滤苏州研究范围外订单。
3. 过滤时间、里程、速度异常订单。
4. 统一时间格式。
5. 生成 dep_datetime、dest_datetime、dep_date、dep_hour。
6. 保留或生成 order_id。
```

### 任务 2：空间离散化

输出：

```text
processed_suzhou/intermediate/area_mapping.csv
```

要求：

```text
1. 选择规则网格或已有区域 polygon。
2. 给每个区域分配连续 node_id：0..n-1。
3. 计算每个节点中心点 center_lng、center_lat。
4. 将每条订单匹配到 origin_node 和 dest_node。
5. 删除无法匹配区域的订单。
```

### 任务 3：生成 OD 矩阵

输出：

```text
processed_suzhou/intermediate/od_count.npy
processed_suzhou/intermediate/od_pairs.csv
```

要求：

```text
1. OD_count[i, j] 表示 origin_node=i、dest_node=j 的订单数。
2. od_pairs.csv 保存非零 OD 对及订单数、平均里程、平均时长。
3. 确保 OD 矩阵维度为 n x n。
```

### 任务 4：生成节点需求 pop

推荐第一版：

```text
pop[i] = 出发订单量[i] + 到达订单量[i]
```

要求：

```text
1. pop 必须为数值。
2. pop 总和必须大于 0。
3. 删除长期零需求且不需要保留的节点。
4. 删除节点后必须重新编号为 0..n-1。
```

### 任务 5：构建 graph.pkl

输出：

```text
processed_suzhou/ppo_swap/suzhou_smoke_1/0/graph.pkl
```

要求：

```text
1. 使用 networkx.Graph。
2. 节点编号为 0..n-1。
3. 每个节点写入 pos=(center_lng, center_lat)。
4. 每个节点写入 pop=pop_value。
5. 每条边写入 length=length_value。
6. length 必须大于 0。
7. 图应连通。
```

### 任务 6：构建 distance_m.pkl

输出：

```text
processed_suzhou/ppo_swap/suzhou_smoke_1/0/distance_m.pkl
```

要求：

```text
1. 保存 n x n 数值矩阵。
2. 行列顺序与 node_id 一致。
3. distance_m[i, j] 表示 i 到 j 的距离。
4. 无 NaN。
5. 无 inf。
6. distance_m.max() > 0。
```

### 任务 7：交付检查

交付前请确认：

```text
1. graph.pkl 可以 pickle.load。
2. distance_m.pkl 可以 pickle.load。
3. graph 节点数 n 与 distance_m shape 一致。
4. 所有节点都有 pop 和 pos。
5. 所有边都有 length。
6. 节点编号为 0..n-1。
7. distance_m 无 NaN/inf。
8. 最大设施数 p 不超过 n。
```

## 11. 第一版统一约定

为避免不同组员各自采用不同口径，第一版苏州数据处理统一采用以下约定。后续如需调整，应先完成第一版可读取、可训练、可评估的闭环，再单独开分支做对比实验。

### 11.1 空间离散化

第一版统一使用规则网格。

要求：

```text
1. 先确定苏州研究范围 bounding box。
2. 使用统一网格边长，例如 1km；如需更细或更粗，必须在处理说明中记录。
3. 上车点和下车点都映射到网格。
4. 有效网格重新编号为连续 node_id：0..n-1。
5. 每个网格节点输出中心点坐标 center_lng、center_lat。
```

第一版不使用行政区、街道、交通小区作为主方案。行政区方案可以保留为后续对比实验。

### 11.2 样本切片

第一版统一生成 3 个样本：

```text
sample 0: 全量样本
sample 1: 早高峰样本
sample 2: 晚高峰样本
```

建议时间定义：

```text
全量样本: 所有通过清洗的订单
早高峰样本: 07:00-10:00
晚高峰样本: 16:00-20:00
```

最终目录名应体现样本数量，例如：

```text
data/suzhou_grid_3/
  0/
    graph.pkl
    distance_m.pkl
  1/
    graph.pkl
    distance_m.pkl
  2/
    graph.pkl
    distance_m.pkl
```

注意：

```text
当前 GraphDataset 会从 data_path 最后一段下划线后的数字读取样本数量。
因此 suzhou_grid_3 表示必须存在 0、1、2 三个样本目录。
```

### 11.3 节点需求 pop

第一版统一定义：

```text
pop[i] = 出发订单量[i] + 到达订单量[i]
```

其中：

```text
出发订单量[i] = sum_j OD_count[i, j]
到达订单量[i] = sum_j OD_count[j, i]
```

要求：

```text
1. pop 必须写入 graph.pkl 的节点属性，字段名固定为 pop。
2. pop 必须是数值。
3. pop 总和必须大于 0。
4. 如果删除零需求网格，必须重新编号 node_id。
```

### 11.4 图边

第一版统一使用相邻网格连边。

建议：

```text
1. 四邻接：上、下、左、右相邻网格连边。
2. 如业务需要，也可使用八邻接，但必须在处理说明中记录。
3. 第一版推荐先用四邻接，减少图结构歧义。
```

每条边必须写入：

```python
G.add_edge(u, v, length=length_value)
```

字段名必须是：

```text
length
```

### 11.5 length 定义

第一版 `length` 统一使用网格中心点距离。

推荐做法：

```text
1. 先把经纬度投影到米制坐标系，例如适用于苏州区域的 UTM 或 CGCS2000/Gauss-Kruger 投影。
2. 在投影坐标中计算两个网格中心点的欧氏距离。
3. 将该距离作为边的 length。
```

可接受的简化做法：

```text
如果已经有网格中心点的米制平面坐标 x/y，可直接用平面欧氏距离。
```

不要直接使用：

```text
经度差
纬度差
经纬度差的平方和
```

原因：

```text
经纬度是角度单位，不是米制距离。
同样的经度差在不同纬度对应的实际距离不同。
直接用经纬度差会导致 length 和 distance_m 的物理含义不稳定。
```

如果暂时无法做投影，至少应使用 haversine 球面距离计算中心点之间的米制距离，并在处理说明中明确记录。


## 12. 最小生成代码模板

```python
import os
import pickle

import networkx as nx
import numpy as np

out_dir = "./data/suzhou_smoke_1/0"
os.makedirs(out_dir, exist_ok=True)

# area_mapping should already provide node_id, center_lng, center_lat, pop
nodes = [
    {"node_id": 0, "center_lng": 120.60, "center_lat": 31.30, "pop": 1000.0},
    {"node_id": 1, "center_lng": 120.61, "center_lat": 31.30, "pop": 800.0},
    {"node_id": 2, "center_lng": 120.60, "center_lat": 31.31, "pop": 1200.0},
]

edges = [
    {"u": 0, "v": 1, "length": 1.0},
    {"u": 0, "v": 2, "length": 1.1},
    {"u": 1, "v": 2, "length": 1.4},
]

G = nx.Graph()

for row in nodes:
    G.add_node(
        row["node_id"],
        pos=(row["center_lng"], row["center_lat"]),
        pop=float(row["pop"]),
    )

for row in edges:
    G.add_edge(row["u"], row["v"], length=float(row["length"]))

n = len(nodes)
shortest = dict(nx.shortest_path_length(G, weight="length"))

distance_m = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        distance_m[i, j] = shortest[i][j]

pickle.dump(G, open(f"{out_dir}/graph.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(distance_m, open(f"{out_dir}/distance_m.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
```
