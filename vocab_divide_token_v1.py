"""
vocab_v3.py
===========
构建 ego / neighbor 专用运动词表，使用两段式 KMeans 聚类。

相比 vocab.py (v2) 的改进
-------------------------
  1. 两段式聚类：MiniBatchKMeans warm-up + 全量 KMeans 精炼
     ── 解决近重复 token 问题
  2. 一站式平衡 balance_segments（静止/运动分离处理）：
        - 运动段：按位移值线性/对数桶分层采样，平衡速度分布
        - 静止段：按精确比例独立采样
     ── 取代旧的 cap_stationary + stratified_subsample（二者会互相打架）
     ── 解决 nbr 词表低速过密、双峰、静止率失控的问题
  3. 自适应 angle_weight：按通道 std 自动归一化
  4. 收紧帧间转向阈值：90° → 30°
     ── 过滤掉物理上不合理的极端噪声 token
  5. 邻居轨迹截断检测：识别中途消失的 nbr，避免末段补零被当成"减速到停"
  6. 向量化 extract_segments：~20-50x 加速
  7. 分块 encode：避免大 N 时 OOM
  8. 自动诊断报告：聚类完即输出质量自检

用法
----
  # ego 专用词表
  python vocab_v3.py \\
      --npz_dir /path/to/cache \\
      --save    vocab/ego_vocab_1024.npz \\
      --vocab_size 1024 --source ego \\
      --angle_weight auto \\
      --max_stationary_ratio 0.05 \\
      --stratify --n_buckets 8

  # nbr 专用词表（强烈建议开启 --stratify）
  python vocab_v3.py \\
      --npz_dir /path/to/cache \\
      --save    vocab/nbr_vocab_1024.npz \\
      --vocab_size 1024 --source nbr \\
      --angle_weight auto \\
      --max_stationary_ratio 0.05 \\
      --stratify --n_buckets 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# ── 全局常量 ─────────────────────────────────────────────────────────────────
HZ         = 10.0
DT_TOKEN   = 0.5
TOKEN_STEP = int(DT_TOKEN * HZ)        # 5 帧/token
T_FUT      = 80
K_TOKENS   = T_FUT // TOKEN_STEP       # 16 tokens
SEG_DIM    = TOKEN_STEP * 3            # 15 = 5帧 × 3维(dx,dy,dh)

PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2
N_SPECIAL = 3

MAX_SPEED_MS  = 40.0
MAX_ANGLE_RAD = np.radians(30.0)       # v2: 90° → v3: 30°（更符合物理约束）

STATIONARY_THRESHOLD = 1.0 * DT_TOKEN  # 0.5m，end_disp 小于此值视为静止


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── 轨迹 → 15维 segments（向量化版本）───────────────────────────────────────

def extract_segments(future_traj: np.ndarray) -> np.ndarray:
    """
    (T, 3) → (K, 15)，每个 segment 存储 5 帧在 token 起点坐标系下的局部位移。
    向量化实现，相比 v2 双重循环约 20-50x 加速。
    """
    T = len(future_traj)
    n_full = T // TOKEN_STEP
    if n_full == 0:
        return np.zeros((0, SEG_DIM), dtype=np.float32)

    traj = future_traj[:n_full * TOKEN_STEP].reshape(n_full, TOKEN_STEP, 3).astype(np.float32)

    # 每段的参考位姿 = 上一段最末帧（第 0 段为原点）
    refs = np.zeros((n_full, 3), dtype=np.float32)
    if n_full > 1:
        refs[1:] = traj[:-1, -1, :]

    cos_h = np.cos(refs[:, 2:3])           # (n_full, 1)
    sin_h = np.sin(refs[:, 2:3])

    dx_g = traj[..., 0] - refs[:, 0:1]     # (n_full, TOKEN_STEP)
    dy_g = traj[..., 1] - refs[:, 1:2]
    dx_l =  cos_h * dx_g + sin_h * dy_g
    dy_l = -sin_h * dx_g + cos_h * dy_g
    dh_l = _wrap_angle(traj[..., 2] - refs[:, 2:3])

    # 交错堆叠为 [dx_0, dy_0, dh_0, dx_1, dy_1, dh_1, ...]
    segs = np.stack([dx_l, dy_l, dh_l], axis=-1).reshape(n_full, SEG_DIM)
    return segs.astype(np.float32)


# ── 数据过滤 ─────────────────────────────────────────────────────────────────

def filter_segments(segments: np.ndarray, verbose: bool = True) -> np.ndarray:
    """过滤超速或异常转向的 segment。"""
    DT_FRAME = 1.0 / HZ
    mask = np.ones(len(segments), dtype=bool)

    speed_killed = 0
    angle_killed = 0

    for j in range(TOKEN_STEP):
        dx_j = segments[:, j * 3 + 0]
        dy_j = segments[:, j * 3 + 1]
        dh_j = segments[:, j * 3 + 2]

        if j == 0:
            step_dx, step_dy, step_dh = dx_j, dy_j, dh_j
        else:
            step_dx = dx_j - segments[:, (j - 1) * 3 + 0]
            step_dy = dy_j - segments[:, (j - 1) * 3 + 1]
            step_dh = _wrap_angle(dh_j - segments[:, (j - 1) * 3 + 2])

        speed_ok = np.sqrt(step_dx**2 + step_dy**2) / DT_FRAME <= MAX_SPEED_MS
        angle_ok = np.abs(step_dh) <= MAX_ANGLE_RAD

        speed_killed += int((mask & ~speed_ok).sum())
        angle_killed += int((mask & ~angle_ok).sum())
        mask &= speed_ok & angle_ok

    result = segments[mask]
    if verbose:
        print(f"[Filter] {len(segments):,} → {len(result):,}  "
              f"（超速 {speed_killed:,} / 超转向 {angle_killed:,}）")
    return result


# ── 静止/运动平衡（合并 cap + stratify，避免互相干扰）──────────────────────

def balance_segments(segments: np.ndarray,
                     max_stationary_ratio: float = 0.05,
                     stratify: bool = True,
                     n_buckets: int = 8,
                     min_motion_disp: float = STATIONARY_THRESHOLD,
                     max_disp: float = 25.0,
                     use_log: bool = True,
                     seed: int = 42,
                     verbose: bool = True) -> np.ndarray:
    """
    一站式平衡静止/运动占比。把两个独立步骤合并以避免互相干扰：
      1) 静止段（end_disp < min_motion_disp）按 max_stationary_ratio 精确采样
      2) 运动段（end_disp >= min_motion_disp）做 stratified 子采样平衡速度分布

    取代 v3 早期版本里 cap_stationary + stratified_subsample 的组合
    （二者会因桶覆盖静止区导致静止率失控）。

    参数
    ----
    max_stationary_ratio : 静止段在最终输出中的精确占比
    stratify             : 是否对运动段做分层采样
    n_buckets            : 运动段分层桶数
    min_motion_disp      : 静止/运动分界（默认 0.5m）
    max_disp             : 分桶上界（米）
    use_log              : 对数桶（推荐，对低速更敏感）/ 线性桶
    """
    if len(segments) == 0:
        return segments

    end_disp = np.sqrt(segments[:, 12]**2 + segments[:, 13]**2)
    is_stat  = end_disp < min_motion_disp
    moving     = segments[~is_stat]
    stationary = segments[is_stat]
    rng = np.random.default_rng(seed)

    # ── 运动段：可选 stratify ────────────────────────────────────────────
    if stratify and len(moving) > 0:
        m_disp = np.sqrt(moving[:, 12]**2 + moving[:, 13]**2)
        if use_log:
            edges = np.geomspace(min_motion_disp, max_disp, n_buckets + 1)
        else:
            edges = np.linspace(min_motion_disp, max_disp, n_buckets + 1)
        edges[0]  = min_motion_disp - 1e-6
        edges[-1] = max(edges[-1], m_disp.max() + 1e-6)
        bucket_ids = np.digitize(m_disp, edges[1:-1])
        counts = np.bincount(bucket_ids, minlength=n_buckets)
        nonzero = counts[counts > 0]
        target = int(np.median(nonzero)) if len(nonzero) else 0

        parts = []
        for b in range(n_buckets):
            idxs = np.where(bucket_ids == b)[0]
            if len(idxs) == 0:
                continue
            n_take = min(len(idxs), target)
            parts.append(moving[rng.choice(idxs, n_take, replace=False)])
        moving_kept = np.concatenate(parts) if parts else moving

        if verbose:
            print(f"[Balance] 运动桶边界 (m): "
                  f"{' '.join(f'{e:.2f}' for e in edges)}")
            print(f"[Balance] 运动桶分布 (原始): "
                  f"{' / '.join(str(c) for c in counts)}")
            print(f"[Balance] 运动: {len(moving):,} → stratify → "
                  f"{len(moving_kept):,}（每桶目标 ~{target:,}）")
    else:
        moving_kept = moving
        if verbose:
            print(f"[Balance] 运动: {len(moving):,}（未启用 stratify）")

    # ── 静止段：按精确比例采样 ──────────────────────────────────────────
    if len(stationary) > 0 and max_stationary_ratio > 0:
        n_stat_target = int(len(moving_kept) * max_stationary_ratio /
                            (1 - max_stationary_ratio))
        n_stat_target = min(n_stat_target, len(stationary))
        kept_stat = stationary[rng.choice(len(stationary),
                                          n_stat_target, replace=False)]
    else:
        kept_stat = stationary[:0]

    result = np.concatenate([moving_kept, kept_stat]) if len(moving_kept) else kept_stat
    actual = len(kept_stat) / len(result) if len(result) else 0
    if verbose:
        print(f"[Balance] 静止: {len(stationary):,} → cap → {len(kept_stat):,}")
        print(f"[Balance] 合计: {len(result):,}  实际静止率: {actual*100:.2f}%")
    return result


# ── 邻居轨迹有效性 ──────────────────────────────────────────────────────────

def _is_valid_nbr_traj(traj: np.ndarray, min_valid_frames: int = TOKEN_STEP * 2) -> tuple:
    """
    检测邻居轨迹的有效区段，截掉中途消失（末段全零）的部分。
    返回 (是否可用, 有效末帧索引+1)。
    """
    valid_per_frame = ~np.all(traj[:, :2] == 0, axis=1)
    if valid_per_frame.sum() < min_valid_frames:
        return False, 0

    # 找到最后一个有效帧
    valid_idxs = np.where(valid_per_frame)[0]
    last_valid = int(valid_idxs[-1]) + 1

    # 截到 TOKEN_STEP 整数倍
    last_valid = (last_valid // TOKEN_STEP) * TOKEN_STEP
    if last_valid < min_valid_frames:
        return False, 0
    return True, last_valid


# ── 数据收集 ─────────────────────────────────────────────────────────────────

def collect_segments(npz_dir: str,
                     max_files: Optional[int] = None,
                     max_stationary_ratio: float = 0.05,
                     stratify: bool = False,
                     n_buckets: int = 8,
                     source: str = 'all',
                     seed: int = 42) -> np.ndarray:
    """
    遍历 .npz 目录，收集 15D segments 并过滤、平衡。

    参数
    ----
    source     : 'all' | 'ego' | 'nbr'
    stratify   : 是否对运动段做分层采样（推荐 nbr 词表开启）
    n_buckets  : 分层桶数
    """
    assert source in ('all', 'ego', 'nbr'), f"source 须为 all/ego/nbr，实际 {source}"
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab] 扫描 {len(files)} 个 .npz 文件（source={source}）...")

    all_segs = []
    n_nbr_total = 0
    n_nbr_skip  = 0

    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"  进度 {i}/{len(files)}")
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
            continue

        if source in ('all', 'ego') and "ego_agent_future" in data:
            s = extract_segments(data["ego_agent_future"])
            if len(s):
                all_segs.append(s)

        if source in ('all', 'nbr') and "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]
            for n in range(nbr.shape[0]):
                n_nbr_total += 1
                ok, end = _is_valid_nbr_traj(nbr[n])
                if not ok:
                    n_nbr_skip += 1
                    continue
                s = extract_segments(nbr[n, :end])
                if len(s):
                    all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未从 {npz_dir} 收集到任何 segments（source={source}）")

    if source in ('all', 'nbr'):
        print(f"[Vocab] 邻居轨迹有效性: 总 {n_nbr_total:,}, 跳过 {n_nbr_skip:,} "
              f"(无效率 {n_nbr_skip/max(n_nbr_total,1)*100:.1f}%)")

    raw = np.concatenate(all_segs, axis=0)
    print(f"[Vocab] 收集完毕（{len(raw):,} 条），开始过滤噪声 ...")
    filtered = filter_segments(raw)

    print(f"[Vocab] 平衡静止/运动占比 → 静止 {max_stationary_ratio*100:.0f}%, "
          f"stratify={stratify} ...")
    filtered = balance_segments(
        filtered,
        max_stationary_ratio = max_stationary_ratio,
        stratify             = stratify,
        n_buckets            = n_buckets,
        seed                 = seed,
    )

    return filtered


# ── MotionVocabulary（两段式 KMeans）────────────────────────────────────────

class MotionVocabulary:
    PAD_IDX   = PAD_IDX
    BOS_IDX   = BOS_IDX
    EOS_IDX   = EOS_IDX
    N_SPECIAL = N_SPECIAL
    SEG_DIM   = SEG_DIM

    def __init__(self,
                 vocab_size: int = 512,
                 angle_weight: Union[float, str] = 3.0,
                 seed: int = 42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight       # float 或 'auto'
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None

    # —— 训练 ——————————————————————————————————————————————————————————————

    def fit(self, segments: np.ndarray,
            batch_size: int = 4096,
            max_iter: int = 300,
            n_init: int = 10,
            refine: bool = True,
            refine_max_samples: int = 200_000):
        """
        两段式聚类：
          Stage 1 — MiniBatchKMeans 在全量数据上 warm-up（n_init=10）
          Stage 2 — 全量 KMeans 在子采样上精炼，初始化用 Stage 1 的中心
        """
        if len(segments) < self.vocab_size:
            raise ValueError(f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")

        # 自适应 angle_weight
        if isinstance(self.angle_weight, str) and self.angle_weight.lower() == 'auto':
            raw_std = segments.std(axis=0).reshape(TOKEN_STEP, 3)
            pos_std = raw_std[:, :2].mean()
            ang_std = raw_std[:, 2].mean()
            auto_w = float(pos_std / (ang_std + 1e-6))
            print(f"[Vocab] auto angle_weight = {auto_w:.3f}  "
                  f"(pos_std={pos_std:.3f}, ang_std={ang_std:.3f})")
            self.angle_weight = auto_w

        X = self._scale(segments)

        # Stage 1: MiniBatchKMeans
        print(f"[Vocab] Stage 1: MiniBatchKMeans warm-up "
              f"({len(X):,} × {SEG_DIM}D → {self.vocab_size}, n_init={n_init}) ...")
        km = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=self.seed,
            n_init=n_init,
            init='k-means++',
            reassignment_ratio=0.01,        # 自动重分配近空簇
            verbose=0,
        )
        km.fit(X)
        print(f"[Vocab] Stage 1 完成。Inertia = {km.inertia_:.4f}")

        if refine:
            # Stage 2: 全量 KMeans 精炼
            rng = np.random.default_rng(self.seed)
            if len(X) > refine_max_samples:
                idx = rng.choice(len(X), refine_max_samples, replace=False)
                X_refine = X[idx]
                print(f"[Vocab] Stage 2: 全量 KMeans 精炼（子采样 {refine_max_samples:,}）...")
            else:
                X_refine = X
                print(f"[Vocab] Stage 2: 全量 KMeans 精炼（{len(X):,}）...")

            km_full = KMeans(
                n_clusters=self.vocab_size,
                init=km.cluster_centers_,
                n_init=1,
                max_iter=100,
                random_state=self.seed,
                tol=1e-5,
            )
            km_full.fit(X_refine)
            centers = km_full.cluster_centers_
            print(f"[Vocab] Stage 2 完成。Inertia = {km_full.inertia_:.4f} "
                  f"(改善 {(km.inertia_ - km_full.inertia_)/km.inertia_*100:+.2f}%)")
        else:
            centers = km.cluster_centers_

        self._centroids = self._unscale(centers).astype(np.float32)
        self._diagnose()
        return self

    # —— 自动诊断 ——————————————————————————————————————————————————————————

    def _diagnose(self):
        """聚类后质量自检报告。"""
        C = self._centroids
        N = len(C)

        # 1. 近重复检测
        diffs = C[:, None] - C[None]
        dists = np.sqrt((diffs**2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        n_dup_strict = int((min_dists < 1e-3).sum())
        n_dup_loose  = int((min_dists < 0.01).sum())

        # 2. 静止 token 占比（按末帧位移，与 balance_segments 一致）
        end_disp = np.sqrt(C[:, 12]**2 + C[:, 13]**2)
        static_rate = float((end_disp < STATIONARY_THRESHOLD).mean())

        # 3. 累积转向覆盖（用末帧 dh，物理意义清晰）
        last_dh = C[:, 14]

        # 4. 累积位移分位数（按各段位移之和）
        seg_disp = np.sqrt(C[:, 0::3]**2 + C[:, 1::3]**2).sum(axis=1)

        print(f"\n[Diag] ====== 词表质量诊断 ======")
        print(f"  vocab_size = {N}, seg_dim = {SEG_DIM}, angle_weight = {self.angle_weight:.3f}")
        print(f"  近重复 token: {n_dup_strict} (dist<1e-3) / {n_dup_loose} (dist<1e-2)")
        print(f"  静止 token 占比 (末段位移<0.5m): {static_rate*100:.1f}%")
        print(f"  累积位移分位数 (m): "
              f"p10={np.percentile(seg_disp,10):.2f}  "
              f"p50={np.percentile(seg_disp,50):.2f}  "
              f"p90={np.percentile(seg_disp,90):.2f}  "
              f"p99={np.percentile(seg_disp,99):.2f}")
        print(f"  末帧累积转向 (rad): "
              f"min={last_dh.min():.2f} ({np.degrees(last_dh.min()):+.0f}°)  "
              f"max={last_dh.max():.2f} ({np.degrees(last_dh.max()):+.0f}°)")
        print(f"  最近邻间距: mean={min_dists.mean():.4f}  median={np.median(min_dists):.4f}")

        warnings = []
        if n_dup_strict > 0:
            warnings.append(f"⚠ {n_dup_strict} 个严重近重复 token（建议增大 n_init 或检查数据）")
        if static_rate > 0.20:
            warnings.append(f"⚠ 静止率 {static_rate*100:.1f}% 偏高（建议降低 max_stationary_ratio）")
        if static_rate < 0.005 and self.vocab_size >= 512:
            warnings.append(f"⚠ 静止率 {static_rate*100:.1f}% 过低，停车场景可能未被覆盖")

        if warnings:
            for w in warnings:
                print(f"  {w}")
        else:
            print(f"  ✓ 未检测到明显异常")
        print(f"[Diag] ============================\n")

    # —— 编码 ——————————————————————————————————————————————————————————————

    def encode(self, segs: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """segs : (N, 15) → token IDs (N,)，含 N_SPECIAL 偏移。分块避免 OOM。"""
        self._check()
        cs = self._scale(self._centroids)
        out = np.empty(len(segs), dtype=np.int64)
        for i in range(0, len(segs), chunk):
            X = self._scale(segs[i:i + chunk])
            diff = X[:, None] - cs[None]
            out[i:i + chunk] = np.argmin(np.linalg.norm(diff, axis=-1), axis=-1)
        return out + self.N_SPECIAL

    def encode_topk(self, segs: np.ndarray, k: int, chunk: int = 8192) -> np.ndarray:
        """segs : (N, 15) → top-k token IDs (N, k)，含 N_SPECIAL 偏移。"""
        self._check()
        cs = self._scale(self._centroids)
        out = np.empty((len(segs), k), dtype=np.int64)
        for i in range(0, len(segs), chunk):
            X = self._scale(segs[i:i + chunk])
            diff = X[:, None] - cs[None]
            dists = np.linalg.norm(diff, axis=-1)
            out[i:i + chunk] = np.argsort(dists, axis=-1)[:, :k]
        return out + self.N_SPECIAL

    # —— I/O ——————————————————————————————————————————————————————————————

    def save(self, path: str):
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 centroids    = self._centroids,
                 vocab_size   = np.int64(self.vocab_size),
                 angle_weight = np.float32(self.angle_weight),
                 seg_dim      = np.int64(SEG_DIM))
        print(f"[Vocab] 已保存 → {path}  (centroids shape: {self._centroids.shape})")

    @classmethod
    def load(cls, path: str):
        d = np.load(path, allow_pickle=False)
        v = cls(vocab_size=int(d["vocab_size"]), angle_weight=float(d["angle_weight"]))
        v._centroids = d["centroids"].astype(np.float32)
        print(f"[Vocab] 已加载 {v.vocab_size}-token 词表（{SEG_DIM}D）← {path}")
        return v

    # —— 内部工具 ——————————————————————————————————————————————————————————

    @property
    def centroids(self) -> np.ndarray:
        self._check()
        return self._centroids

    def _scale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy().astype(np.float32)
        s[:, 2::3] *= float(self.angle_weight)
        return s

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy()
        s[:, 2::3] /= float(self.angle_weight)
        return s

    def _check(self):
        if self._centroids is None:
            raise RuntimeError("词表未初始化，请先调用 fit() 或 load()")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_angle_weight(v: str) -> Union[float, str]:
    if v.lower() == 'auto':
        return 'auto'
    return float(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建运动词表 v3（15D 子轨迹，两段式 KMeans + 静止/运动平衡）"
    )
    parser.add_argument("--npz_dir",    required=True)
    parser.add_argument("--save",       default="./vocab/vocab_512.npz")
    parser.add_argument("--vocab_size", type=int,   default=512)
    parser.add_argument("--max_files",  type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=4096)
    parser.add_argument("--max_iter",   type=int,   default=300)
    parser.add_argument("--n_init",     type=int,   default=10,
                        help="MiniBatchKMeans 重启次数（v2 默认 3）")
    parser.add_argument("--no_refine",  action="store_true",
                        help="关闭 Stage 2 精炼（仅用 MiniBatchKMeans）")
    parser.add_argument("--refine_max_samples", type=int, default=200_000,
                        help="Stage 2 精炼时的最大样本数")
    parser.add_argument("--angle_weight", type=_parse_angle_weight, default=3.0,
                        help="角度权重，可填浮点数或 'auto'（按 std 自适应）")
    parser.add_argument("--max_stationary_ratio", type=float, default=0.05,
                        help="静止 segment 在最终输出中的精确占比，默认 0.05")
    parser.add_argument("--stratify", action="store_true",
                        help="启用分层采样平衡运动段速度分布（推荐 nbr 词表）")
    parser.add_argument("--n_buckets", type=int, default=8,
                        help="运动段分层采样桶数")
    parser.add_argument("--source", choices=['all', 'ego', 'nbr'], default='all')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    segments = collect_segments(
        args.npz_dir, args.max_files,
        max_stationary_ratio = args.max_stationary_ratio,
        stratify             = args.stratify,
        n_buckets            = args.n_buckets,
        source               = args.source,
        seed                 = args.seed,
    )

    vocab = MotionVocabulary(
        vocab_size   = args.vocab_size,
        angle_weight = args.angle_weight,
        seed         = args.seed,
    )
    vocab.fit(
        segments,
        batch_size         = args.batch_size,
        max_iter           = args.max_iter,
        n_init             = args.n_init,
        refine             = not args.no_refine,
        refine_max_samples = args.refine_max_samples,
    )
    vocab.save(args.save)
