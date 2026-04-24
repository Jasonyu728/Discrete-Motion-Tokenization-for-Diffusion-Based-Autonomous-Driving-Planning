

## 一、任务背景

这个项目的目标是**自动驾驶运动规划**：给定当前车辆周围的环境信息（其他车辆、车道线、路线），让模型预测自车（ego）未来 8 秒的行驶轨迹。

评估平台采用 **nuPlan**，它会在仿真环境中让模型实际驾驶，并进行打分。核心指标包括：

* **Score（综合分）**：目标接近 100。
* **Collisions（碰撞）**：是否发生碰撞。
* **Drivable（可行驶区域）**：是否开出可行驶区域。
* **Comfort（舒适度）**：加速度与加加速度是否超标。
* **Progress（进度）**：沿预定路线前进的比例。
* **Making（推进目标）**：是否在推进目标（作为乘数，影响最大）。

> **Score 公式：**
> **Score** = Collisions × Drivable × Direction × Making × 加权均值(TTC, Comfort, Progress, SpeedLimit)
> *(注：任何乘数项为 0，则总 Score = 0)*

---

## 二、原始模型：DiffusionPlanner

基础模型是已有的开源工作 **DiffusionPlanner**。原版使用连续轨迹坐标作为扩散目标，直接在 `(x, y, heading)` 空间进行加噪和去噪。

**核心处理流程：**
1. **环境上下文**（车道 / 车辆 / 路线）
2. ⬇️ 经过 **Encoder**（Transformer）提取上下文特征
3. ⬇️ 送入 **DiT**（扩散模型），从噪声中逐步去噪以预测未来轨迹
4. ⬇️ 输出 **连续轨迹**：形状为 `(B, 80帧, 4维)`，即 `[x, y, cos_h, sin_h]`

---

## 三、我们的核心改动：Token 化轨迹表示

**原版的痛点**：连续轨迹空间维度高且不紧凑，导致扩散模型难以学习到有意义的运动模式。
**改进思路**：借鉴语言模型的思路，将连续轨迹离散化为 Token 序列。

### 3.1 构建运动词表（Vocabulary）
将所有训练场景的轨迹按每 `0.5s`（5 帧）切分为一段，收集海量“运动片段”。使用 `MiniBatchKMeans` 进行聚类，将相似的运动模式归为一类，每一类的中心即为一个 **Token（运动原语）**。
* **Ego 词表**：512 或 1024 个 Token（目前在探索 1024）。
* **Neighbor 词表**：1024 个 Token。
* **Token 维度**：每个 Token 是一个 15 维向量（5 帧 × 3 维：`dx, dy, dh`）。
* **静止率控制**：过滤掉过多的“不动” Token（初始为 15%，后续优化为 5%）。

### 3.2 Tokenize 训练数据
利用词表将每条训练轨迹编码为 16 个 Token ID 的序列（16 × 0.5s = 8s）。
* **序列格式**：`[BOS, t1, t2, ..., t16, EOS]`

### 3.3 构建 Token Embedding 表
每个 Token ID 对应一个 `D` 维的 Embedding 向量（目前 $D=64$，探索 $D=128$）。
* **生成方式**：`centroid (15维) × 随机投影矩阵 (15×D) → embedding (D维)`
* **特性**：Embedding 表固定不训练，作为“运动含义的坐标系”。

### 3.4 训练：MSE Loss in Embedding Space
数据在 Embedding 空间的训练前向传递过程如下：

1. **GT token IDs** 查找 Embedding 表得到 $x_0$ `(B, P, 16×D)`
2. ⬇️ 经过 **VPSDE** 加噪得到 $x_t$
3. ⬇️ 经过 **DiT** 去噪
4. ⬇️ 输出预测值 **pred** `(B, P, 16×D)`

   
**损失函数：**
$$Loss = \alpha \times MSE(pred_{ego}, x_{0\_ego}) + MSE(pred_{nbr}, x_{0\_nbr})$$

*(注：$\alpha$ 即 `alpha_planning_loss`，用于控制 ego 和 neighbor 的损失权重比，通常设为 3.0 或 5.0)*

### 3.5 推理：Embedding → Token → 轨迹
1. **DiT 输出 pred** `(B, P, 16×D)`
2. ⬇️ 最近邻查找 (`cdist`)
3. **Token IDs** `(B, 16)`
4. ⬇️ 经过 **TokenTrajectoryDecoder**（逐帧展开 15 维运动片段，拼成完整轨迹，并添加高斯平滑以消除 Token 边界的速度跳变）
5. **连续轨迹** `(B, 80帧, 4维)`

---

## 四、完整数据流

以下为项目从原始数据到最终评估结果的完整数据流向：

```text
nuPlan 原始数据
    ↓  (data_process.py)
.npz 格式文件 (每个场景一个文件，含车辆状态、车道、路线、未来轨迹)
    ↓  (vocab_divide_token.py - K-Means 聚类)
ego_vocab_1024.npz + nbr_vocab_1024.npz
    ↓  (tokenize_npz.py)
每个 .npz 文件新增 ego_token_ids / neighbor_token_ids 字段
    ↓  (torch_run.sh → train_predictor.py → train_epoch.py)
训练好的模型 Checkpoint (latest.pth + args.json)
    ↓  (sim_diffusion_planner_runner.sh → nuPlan 仿真)
评估结果 (.parquet 文件)
    ↓  (read_eval_results.py)
all_results.csv (包含 Score, Collisions, TTC, Drivable, Comfort, Progress 等指标)
```
