"""
vocabulary_v2.py
================
在 vocabulary_filter.py 基础上改进的词表构建脚本。

改进目标
--------
large 版本（原始 K-Means，无分层）评估结果：
  drivable_area=100%, making_progress=8.33%
根因：数据集中 60%+ 是静止 segment → 词表以静止 token 为主 → 模型学到"不动最安全"

改进策略：1D 速度分层，但各区间目标数不同：
  - 静止区间：大幅削减（10k），避免主导词表
  - 低速/慢速/中速直行：大幅增加（40k），城市正常行驶的主体
  - 快速/高速：适量保留（20-30k）
  - 不引入转向分层（2D 版本急弯过多导致 driving_direction=0%）

用法
----
  python vocabulary_v2.py \
      --npz_dir /path/to/nuplan_diffusionplanner_large \
      --save ./npz2token_dataset/vocab_512_v2.npz \
      --vocab_size 512
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ── 全局常量 ──────────────────────────────────────────────────────────────
HZ         = 10.0
DT_TOKEN   = 0.5
TOKEN_STEP = int(DT_TOKEN * HZ)   # = 5
T_FUT      = 80
K_TOKENS   = T_FUT // TOKEN_STEP  # = 16

PAD_IDX   = 0
BOS_IDX   = 1
EOS_IDX   = 2
N_SPECIAL = 3

MAX_SPEED_MS  = 40.0
MAX_ANGLE_RAD = np.radians(90.0)

# ── 各速度区间的采样目标数 ────────────────────────────────────────────────
# 顺序对应：[静止, 低速, 慢速, 中速, 快速, 高速]
# 设计逻辑：
#   静止(10k)  : large 版本静止占 60%+ → 削减到 ~5%，避免词表以静止为主
#   低速(40k)  : 城市停走、低速通行，最常见的有效运动
#   慢速(40k)  : 正常城市行驶
#   中速(40k)  : 干道行驶
#   快速(30k)  : 快速路/匝道
#   高速(20k)  : 高速公路，相对少见
BIN_TARGETS = [10_000, 40_000, 40_000, 40_000, 30_000, 20_000]


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def extract_segments(future_traj: np.ndarray) -> np.ndarray:
    """(T,3) [x,y,heading] → (K,3) [dx_local, dy_local, dheading]"""
    T = len(future_traj)
    segs = []
    x_ref, y_ref, h_ref = 0.0, 0.0, 0.0
    for i in range(K_TOKENS):
        end_idx = (i + 1) * TOKEN_STEP - 1
        if end_idx >= T:
            break
        x_end, y_end, h_end = future_traj[end_idx]
        dx_g = x_end - x_ref
        dy_g = y_end - y_ref
        cos_h, sin_h = np.cos(h_ref), np.sin(h_ref)
        dx_l =  cos_h * dx_g + sin_h * dy_g
        dy_l = -sin_h * dx_g + cos_h * dy_g
        dh   = float(_wrap_angle(np.array([h_end - h_ref]))[0])
        segs.append([dx_l, dy_l, dh])
        x_ref, y_ref, h_ref = x_end, y_end, h_end
    return np.array(segs, dtype=np.float32) if segs else np.zeros((0, 3), dtype=np.float32)


def filter_segments(segments: np.ndarray) -> np.ndarray:
    """过滤噪声：超速(>144km/h)和异常转向(>90°/0.5s)。"""
    n_before = len(segments)
    speed = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2) / DT_TOKEN
    mask_speed = speed <= MAX_SPEED_MS
    mask_angle = np.abs(segments[:, 2]) <= MAX_ANGLE_RAD
    mask = mask_speed & mask_angle
    result = segments[mask]
    n_after = len(result)

    print(f"[Filter] 原始 segments        : {n_before:,}")
    print(f"[Filter] 超速异常 (>144 km/h) : -{(~mask_speed).sum()}")
    print(f"[Filter] 转向异常 (>90°/0.5s) : -{(~mask_angle).sum()}")
    print(f"[Filter] 过滤后保留           : {n_after:,}  ({n_after/n_before*100:.1f}%)")

    # 速度分布
    spd = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2) / DT_TOKEN
    spd_bins   = [0, 1, 3, 6, 10, 20, MAX_SPEED_MS + 1]
    spd_labels = ['静止  <1 m/s ', '低速 1-3 m/s', '慢速 3-6 m/s',
                  '中速 6-10m/s', '快速10-20m/s', '高速20-40m/s']
    print("[Filter] 过滤后速度分布:")
    for i, label in enumerate(spd_labels):
        cnt = int(((spd >= spd_bins[i]) & (spd < spd_bins[i+1])).sum())
        pct = cnt / n_after * 100 if n_after else 0
        bar = '█' * min(cnt * 40 // max(n_after, 1), 40)
        print(f"  {label}: {cnt:7,d}  ({pct:5.1f}%)  {bar}")

    return result


def stratified_sample_v2(segments: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    按速度区间分层采样，各区间目标数按驾驶场景频率加权。

    静止区间大幅削减（避免 large 版本的 making_progress 崩溃问题），
    低中速前向运动区间增加（城市行驶主体），
    不引入转向分层（避免 2D 版本的 driving_direction 崩溃问题）。
    """
    rng = np.random.default_rng(seed)
    speed = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)

    bins   = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, np.inf]
    labels = ['静止  <0.5m', '低速 0.5-1m', '慢速 1-2.5m',
              '中速 2.5-5m', '快速  5-10m', '高速  >10m ']

    sampled = []
    total_target = sum(BIN_TARGETS)
    print("[Stratify-v2] 分层采样统计（目标数按驾驶频率加权）:")
    print(f"  {'区间':12s}  {'可用':>10s}  {'目标':>8s}  {'采样':>8s}  {'占比':>6s}")

    for i, label in enumerate(labels):
        mask     = (speed >= bins[i]) & (speed < bins[i + 1])
        idx      = np.where(mask)[0]
        n_avail  = len(idx)
        n_sample = min(n_avail, BIN_TARGETS[i])
        chosen   = rng.choice(idx, size=n_sample, replace=False)
        sampled.append(segments[chosen])
        pct = n_sample / total_target * 100
        print(f"  {label:12s}  {n_avail:>10,d}  {BIN_TARGETS[i]:>8,d}  {n_sample:>8,d}  {pct:>5.1f}%")

    result = np.concatenate(sampled, axis=0)
    rng.shuffle(result)
    print(f"[Stratify-v2] 采样后总量: {len(result):,}  (原始: {len(segments):,})")
    return result


def collect_segments(npz_dir: str, max_files=None) -> np.ndarray:
    """遍历 .npz 目录，收集所有 ego + neighbor 的 segments 并过滤噪声。"""
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab] 扫描 {len(files)} 个 .npz 文件 ...")

    all_segs = []
    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"  进度 {i}/{len(files)}")
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
            continue

        if "ego_agent_future" in data:
            s = extract_segments(data["ego_agent_future"])
            if len(s):
                all_segs.append(s)

        if "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]
            for n in range(nbr.shape[0]):
                if np.allclose(nbr[n, :TOKEN_STEP, :2], 0):
                    continue
                s = extract_segments(nbr[n])
                if len(s):
                    all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未能从 {npz_dir} 收集到任何 segments")

    raw = np.concatenate(all_segs, axis=0)
    print("[Vocab] 收集完毕，开始过滤噪声 ...")
    return filter_segments(raw)


# ── MotionVocabulary ──────────────────────────────────────────────────────

class MotionVocabulary:
    PAD_IDX   = PAD_IDX
    BOS_IDX   = BOS_IDX
    EOS_IDX   = EOS_IDX
    N_SPECIAL = N_SPECIAL

    def __init__(self, vocab_size=512, angle_weight=3.0, seed=42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None
        self._predictor = None

    def fit(self, segments: np.ndarray, batch_size=20_000):
        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(f"segments 应为 (N,3)，实际 {segments.shape}")
        if len(segments) < self.vocab_size:
            raise ValueError(
                f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")
        X = self._scale(segments)
        print(f"[Vocab] k-means: {len(X):,} segments → {self.vocab_size} 聚类 ...")
        km = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=min(batch_size, len(X)),
            n_init=3, max_iter=300,
            random_state=self.seed, verbose=0)
        km.fit(X)
        self._centroids = self._unscale(km.cluster_centers_).astype(np.float32)
        self._predictor = _Predictor(self._centroids, self.angle_weight)
        print(f"[Vocab] 完成。Inertia = {km.inertia_:.4f}")
        return self

    def save(self, path: str):
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 centroids    = self._centroids,
                 vocab_size   = np.int64(self.vocab_size),
                 angle_weight = np.float32(self.angle_weight))
        print(f"[Vocab] 已保存 → {path}")

    @classmethod
    def load(cls, path: str):
        d = np.load(path, allow_pickle=False)
        v = cls(vocab_size=int(d["vocab_size"]), angle_weight=float(d["angle_weight"]))
        v._centroids = d["centroids"].astype(np.float32)
        v._predictor = _Predictor(v._centroids, v.angle_weight)
        print(f"[Vocab] 已加载 {v.vocab_size}-token 词表 ← {path}")
        return v

    def _scale(self, x):
        s = x.copy().astype(np.float32)
        s[:, 2] *= self.angle_weight
        return s

    def _unscale(self, x):
        s = x.copy()
        s[:, 2] /= self.angle_weight
        return s

    def _check(self):
        if self._centroids is None:
            raise RuntimeError("词表未初始化，请先调用 fit() 或 load()")


class _Predictor:
    def __init__(self, centroids, angle_weight):
        self._cs = centroids.copy().astype(np.float32)
        self._cs[:, 2] *= angle_weight

    def predict(self, Xs):
        diff  = Xs[:, None] - self._cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        return np.argmin(dists, axis=-1)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 v2 运动词表（速度加权分层采样）")
    parser.add_argument("--npz_dir",    required=True,  help="npz 数据目录")
    parser.add_argument("--save",       default="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/npz2token_dataset/vocab_512_v2.npz")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_files",  type=int, default=None)
    args = parser.parse_args()

    segments = collect_segments(args.npz_dir, args.max_files)
    segments = stratified_sample_v2(segments)

    vocab = MotionVocabulary(vocab_size=args.vocab_size)
    vocab.fit(segments)
    vocab.save(args.save)
