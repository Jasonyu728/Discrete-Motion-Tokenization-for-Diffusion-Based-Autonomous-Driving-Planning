"""
vocabulary_v4.py
================
最基本的 K-Medoids 词表构建：数据过滤 + K-Medoids 聚类，不做任何采样分层。

相比各版本
----------
  large (vocabulary_filter.py) : 过滤 + K-Means，无分层
  v2                           : 过滤 + 速度加权分层 + K-Means
  v3                           : 过滤 + 速度加权分层 + K-Medoids
  v4 (本文件)                  : 过滤 + K-Medoids，无分层  ← 最基本

用法
----
  python vocabulary_v4.py \
      --npz_dir /path/to/nuplan_diffusionplanner_large \
      --save ./npz2token_dataset/vocab_512_v4.npz \
      --vocab_size 512
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn_extra.cluster import KMedoids

# ── 全局常量 ──────────────────────────────────────────────────────────────
HZ         = 10.0
DT_TOKEN   = 0.5
TOKEN_STEP = int(DT_TOKEN * HZ)
T_FUT      = 80
K_TOKENS   = T_FUT // TOKEN_STEP

PAD_IDX   = 0
BOS_IDX   = 1
EOS_IDX   = 2
N_SPECIAL = 3

MAX_SPEED_MS  = 40.0
MAX_ANGLE_RAD = np.radians(90.0)


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

    def fit(self, segments: np.ndarray, max_samples: int = 30_000):
        """
        K-Medoids 聚类。
        centroid 保证是数据集中真实存在的 segment，不是均值。

        K-Medoids 需要计算 N×N 距离矩阵（内存为 N²×4 字节），
        因此必须先随机子采样到 max_samples 再聚类：
          30k 样本 → 距离矩阵 ~3.6 GB，可行
          12.7M 样本 → 距离矩阵 ~586 TiB，不可能
        """
        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(f"segments 应为 (N,3)，实际 {segments.shape}")
        if len(segments) < self.vocab_size:
            raise ValueError(
                f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")

        # 随机子采样
        if len(segments) > max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(segments), size=max_samples, replace=False)
            segments = segments[idx]
            print(f"[Vocab] 随机子采样: {len(idx):,} / {len(segments):,} 条用于 K-Medoids")

        X = self._scale(segments)
        print(f"[Vocab] K-Medoids: {len(X):,} segments → {self.vocab_size} 聚类 ...")
        print(f"[Vocab] 使用 alternate 方法，请耐心等待 ...")

        km = KMedoids(
            n_clusters=self.vocab_size,
            metric='euclidean',
            method='alternate',
            init='k-medoids++',
            max_iter=300,
            random_state=self.seed
        )
        km.fit(X)

        self._centroids = self._unscale(X[km.medoid_indices_]).astype(np.float32)
        self._predictor = _Predictor(self._centroids, self.angle_weight)
        print(f"[Vocab] 完成。Inertia = {km.inertia_:.4f}")
        print(f"[Vocab] 所有 centroid 均为真实数据点（K-Medoids 保证）")
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
    parser = argparse.ArgumentParser(description="构建 v4 运动词表（过滤 + K-Medoids，无分层）")
    parser.add_argument("--npz_dir",    required=True,  help="npz 数据目录")
    parser.add_argument("--save",       default="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/npz2token_dataset/vocab_512_v4.npz")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_files",  type=int, default=None)
    args = parser.parse_args()

    segments = collect_segments(args.npz_dir, args.max_files)

    vocab = MotionVocabulary(vocab_size=args.vocab_size)
    vocab.fit(segments)
    vocab.save(args.save)
