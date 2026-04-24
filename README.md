一、任务背景
这个项目的目标是自动驾驶运动规划：给定当前车辆周围的环境信息（其他车辆、车道线、路线），让模型预测自车（ego）未来 8 秒的行驶轨迹。
评估平台是 nuPlan，它会在仿真环境中让模型实际开车，然后打分。核心指标包括：
● Score：综合分（目标接近100）
● Collisions：不碰撞
● Drivable：不开出可行驶区域
● Comfort：加速度/加加速度不超标
● Progress：沿预定路线前进的比例
● Making：是否在推进目标（乘数，影响最大）
Score 公式：Score = Collisions × Drivable × Direction × Making × 加权均值(TTC, Comfort, Progress, SpeedLimit)，任何乘数项为 0 则 Score=0。

二、原始模型：DiffusionPlanner
基础模型是 DiffusionPlanner（已有开源工作）。核心思路：
环境上下文（车道/车辆/路线）
        ↓  Encoder（Transformer）
        上下文特征
        ↓  DiT（扩散模型）
        从噪声逐步去噪，预测未来轨迹
        ↓
  连续轨迹 (B, 80帧, 4维) [x, y, cos_h, sin_h]
原版用连续轨迹坐标作为扩散目标，直接在 (x, y, heading) 空间加噪/去噪。

三、我们的核心改动：Token 化轨迹表示
原版的问题：连续轨迹空间很高维、不紧凑，扩散模型难以学到有意义的运动模式。
我们的做法：借鉴语言模型的思路，把连续轨迹离散化成 token 序列。
3.1 构建运动词表（Vocabulary）
把所有训练场景的轨迹切成每 0.5s 一段（5帧），收集海量"运动片段"，用 MiniBatchKMeans 聚类，把相似的运动模式归为一类，每类的中心叫做一个 token（运动原语）。
● Ego 词表：512 或 1024 个 token（现在在探索 1024）
● Neighbor 词表：1024 个 token
● 每个 token 是一个 15 维向量（5帧 × 3维：dx, dy, dh）
● 静止率控制：过滤掉过多的"不动"token，初始 15%，后来改成 5%
3.2 Tokenize 训练数据
用词表把每条训练轨迹编码成 16 个 token ID 的序列（16 × 0.5s = 8s），格式：[BOS, t1, t2, ..., t16, EOS]
3.3 构建 Token Embedding 表
每个 token ID 对应一个 D 维 embedding 向量（目前 D=64，探索 D=128）：
centroid (15维) × 随机投影矩阵 (15×D) → embedding (D维)
Embedding 表固定不训练，作为"运动含义的坐标系"。
3.4 训练：MSE Loss in Embedding Space
GT token IDs → 查 embedding 表 → x0 (B, P, 16×D)
                                       ↓ VPSDE 加噪
                                      xt
                                       ↓ DiT 去噪
                                  pred (B, P, 16×D)

Loss = α × MSE(pred_ego, x0_ego) + MSE(pred_nbr, x0_nbr)
α（alpha_planning_loss）控制 ego 和 neighbor 的损失权重比，通常设为 3.0 或 5.0。
3.5 推理：Embedding → Token → 轨迹
DiT 输出 pred (B, P, 16×D)
    ↓  最近邻查找（cdist）
Token IDs (B, 16)
    ↓  TokenTrajectoryDecoder
连续轨迹 (B, 80帧, 4维)
TokenTrajectoryDecoder 把 token ID 对应的 centroid（15维运动片段）逐帧展开，拼成完整轨迹，最后加一步高斯平滑消除 token 边界的速度跳变。

四、完整数据流
nuPlan 原始数据
    ↓ data_process.py
.npz 格式（每个场景一个文件，含车辆状态、车道、路线、未来轨迹）
    ↓ vocab_divide_token.py（k-means 聚类）
ego_vocab_1024.npz + nbr_vocab_1024.npz
    ↓ tokenize_npz.py
每个 .npz 新增 ego_token_ids / neighbor_token_ids 字段
    ↓ torch_run.sh → train_predictor.py → train_epoch.py
训练好的模型 checkpoint（latest.pth + args.json）
    ↓ sim_diffusion_planner_runner.sh → nuPlan 仿真
评估结果（parquet 文件）
    ↓ read_eval_results.py
all_results.csv（Score, Collisions, TTC, Drivable, Comfort, Progress...）

