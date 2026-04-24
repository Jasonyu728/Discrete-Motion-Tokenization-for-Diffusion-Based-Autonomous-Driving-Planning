"""
vocabulary.py
=============
为 DiffusionPlanner 真实数据格式构建 512-token 运动词表。

真实 .npz 字段格式（已确认）
-----------------------------
  ego_agent_future        : (80, 3)      [x, y, heading]  ego局部坐标系, 8s@10Hz
  neighbor_agents_future  : (32, 80, 3)  同上，每个agent一行

坐标系说明
----------
  - 所有轨迹已转换到 ego 在当前时刻的局部坐标系
  - ego 当前位置 = (0, 0), 朝向 = 0
  - neighbor 的坐标也在这个系里

Token 粒度
----------
  DT_TOKEN = 0.5s,  TOKEN_STEP = 5 帧,  K = 80/5 = 16 个token / 场景

用法
----
  # 第一步：构建词表（只做一次）
  python vocabulary.py \
      --npz_dir  /path/to/cache \
      --save     ./vocab_512.npz \
      --vocab_size 512

  # 后续使用
  from vocabulary import MotionVocabulary
  vocab = MotionVocabulary.load("vocab_512.npz")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ── 全局常量 ─────────────────────────────────────────────────────────────
HZ         = 10.0          # nuPlan 采样率
DT_TOKEN   = 0.5           # 每个 token 覆盖 0.5 秒
TOKEN_STEP = int(DT_TOKEN * HZ)   # = 5 帧
T_FUT      = 80            # 未来帧数
K_TOKENS   = T_FUT // TOKEN_STEP  # = 16 个 motion token / 场景

# 特殊 token ID
PAD_IDX  = 0
BOS_IDX  = 1
EOS_IDX  = 2
N_SPECIAL = 3


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """将角度包裹到 [-π, π]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def extract_segments(future_traj: np.ndarray) -> np.ndarray:
    """
    从一条未来轨迹中提取位移段（displacement segments）。

    Parameters
    ----------
    future_traj : (T, 3)  列为 [x, y, heading]，ego 局部坐标系
                  T 必须 >= TOKEN_STEP

    Returns
    -------
    segments : (K, 3)  每段 [dx_local, dy_local, dheading]
               dx_local / dy_local 是在**段起点局部坐标系**下的位移
    """
    T = len(future_traj)
    # 当前时刻参考状态（ego 原点）
    refs = [(0.0, 0.0, 0.0)]   # (x_ref, y_ref, h_ref) 列表，用于 rolling 参考

    # 预先计算每个 token 边界的 GT 状态
    # token i 的终点帧索引 = (i+1)*TOKEN_STEP - 1
    segs = []
    x_ref, y_ref, h_ref = 0.0, 0.0, 0.0   # t=0 时 ego 原点

    for i in range(K_TOKENS):
        end_idx = (i + 1) * TOKEN_STEP - 1
        if end_idx >= T:
            break

        x_end, y_end, h_end = future_traj[end_idx]

        # 将全局（ego坐标系）位移转到段起点局部坐标系
        dx_g = x_end - x_ref
        dy_g = y_end - y_ref
        cos_h = np.cos(h_ref)
        sin_h = np.sin(h_ref)
        dx_l =  cos_h * dx_g + sin_h * dy_g
        dy_l = -sin_h * dx_g + cos_h * dy_g
        dh   = float(_wrap_angle(np.array([h_end - h_ref]))[0])

        segs.append([dx_l, dy_l, dh])

        # 更新参考点（GT rolling：用 GT 终点作为下一段的参考）
        x_ref, y_ref, h_ref = x_end, y_end, h_end

    return np.array(segs, dtype=np.float32) if segs else np.zeros((0, 3), dtype=np.float32)


# ── MotionVocabulary ──────────────────────────────────────────────────────

class MotionVocabulary:
    """
    512-token 运动词表。
    codebook 中每个 token 是 [dx_local, dy_local, dheading] 的聚类中心。
    token ID 0/1/2 为特殊 token（PAD/BOS/EOS），motion token 从 3 开始。
    """

    PAD_IDX  = PAD_IDX
    BOS_IDX  = BOS_IDX
    EOS_IDX  = EOS_IDX
    N_SPECIAL = N_SPECIAL

    def __init__(self, vocab_size: int = 512, angle_weight: float = 3.0, seed: int = 42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None
        self._predictor = None

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, segments: np.ndarray, batch_size: int = 20_000) -> "MotionVocabulary":
        """
        Parameters
        ----------
        segments : (N, 3)  所有场景的 displacement segments 的集合
        """
        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(f"segments 应为 (N,3)，实际得到 {segments.shape}")

        X = self._scale(segments)
        print(f"[Vocab] 开始 k-means: {len(X):,} 个 segments → {self.vocab_size} 个聚类 ...")
        km = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=min(batch_size, len(X)),
            n_init=3, max_iter=300,
            random_state=self.seed, verbose=0,
        )
        km.fit(X)
        self._centroids = self._unscale(km.cluster_centers_).astype(np.float32)
        self._predictor  = _Predictor(self._centroids, self.angle_weight)
        print(f"[Vocab] 完成。Inertia = {km.inertia_:.4f}")
        return self

    # ── encode ───────────────────────────────────────────────────────────

    def encode(self, segments: np.ndarray) -> np.ndarray:
        """(N, 3) → (N,) int64，值在 [N_SPECIAL, N_SPECIAL+vocab_size)"""
        self._check()
        raw = self._predictor.predict(self._scale(segments))
        return (raw + self.N_SPECIAL).astype(np.int64)

    def encode_topk(self, segments: np.ndarray, k: int = 5) -> np.ndarray:
        """(N, 3) → (N, k) int64，按距离从近到远排列"""
        self._check()
        Xs = self._scale(segments)
        Cs = self._scale(self._centroids)
        diff  = Xs[:, None] - Cs[None]           # (N, V, 3)
        dists = np.linalg.norm(diff, axis=-1)     # (N, V)
        topk  = np.argsort(dists, axis=-1)[:, :k] # (N, k)
        return (topk + self.N_SPECIAL).astype(np.int64)

    # ── decode ───────────────────────────────────────────────────────────

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """(...,) int64 → (..., 3) float32，特殊 token 映射到零向量"""
        self._check()
        flat = np.asarray(indices, dtype=np.int64).ravel()
        out  = np.zeros((len(flat), 3), dtype=np.float32)
        ok   = flat >= self.N_SPECIAL
        if ok.any():
            raw = (flat[ok] - self.N_SPECIAL).clip(0, self.vocab_size - 1)
            out[ok] = self._centroids[raw]
        return out.reshape(indices.shape + (3,))

    # ── save / load ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 centroids    = self._centroids,
                 vocab_size   = np.int64(self.vocab_size),
                 angle_weight = np.float32(self.angle_weight))
        print(f"[Vocab] 已保存 → {path}")

    @classmethod
    def load(cls, path: str) -> "MotionVocabulary":
        d = np.load(path, allow_pickle=False)
        v = cls(vocab_size   = int(d["vocab_size"]),
                angle_weight = float(d["angle_weight"]))
        v._centroids = d["centroids"].astype(np.float32)
        v._predictor = _Predictor(v._centroids, v.angle_weight)
        print(f"[Vocab] 已加载 {v.vocab_size}-token 词表 ← {path}")
        return v

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size + self.N_SPECIAL

    @property
    def centroids(self) -> np.ndarray:
        self._check()
        return self._centroids

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _scale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy().astype(np.float32)
        s[:, 2] *= self.angle_weight
        return s

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy()
        s[:, 2] /= self.angle_weight
        return s

    def _check(self) -> None:
        if self._centroids is None:
            raise RuntimeError("词表未初始化，请先调用 fit() 或 load()")


class _Predictor:
    """最近邻查找（不依赖 sklearn）"""
    def __init__(self, centroids: np.ndarray, angle_weight: float):
        self._cs = centroids.copy().astype(np.float32)
        self._cs[:, 2] *= angle_weight

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        diff  = Xs[:, None] - self._cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        return np.argmin(dists, axis=-1)


# ── 从 npz 目录收集 segments ──────────────────────────────────────────────

def collect_segments(npz_dir: str, max_files: Optional[int] = None) -> np.ndarray:
    """
    遍历 DiffusionPlanner 的 .npz 文件，收集所有 ego + neighbor 的
    displacement segments，用于词表拟合。
    """
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab] 扫描 {len(files)} 个 .npz 文件 ...")

    all_segs = []
    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"  进度 {i}/{len(files)}，已收集 {sum(len(s) for s in all_segs):,} 段")
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
            continue

        # ego 未来轨迹
        if "ego_agent_future" in data:
            segs = extract_segments(data["ego_agent_future"])
            if len(segs) > 0:
                all_segs.append(segs)

        # neighbor 未来轨迹
        if "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]   # (32, 80, 3)
            for n in range(nbr.shape[0]):
                # 跳过全零（不存在的 agent）
                if np.allclose(nbr[n, :TOKEN_STEP, :2], 0):
                    continue
                segs = extract_segments(nbr[n])
                if len(segs) > 0:
                    all_segs.append(segs)

    if not all_segs:
        raise RuntimeError(f"未能从 {npz_dir} 收集到任何 segments，请检查路径和文件格式")

    result = np.concatenate(all_segs, axis=0)
    print(f"[Vocab] 共收集 {len(result):,} 个 displacement segments")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建运动词表")
    parser.add_argument("--npz_dir",    required=True, help="DiffusionPlanner cache 目录")
    parser.add_argument("--save",       default="vocab_512.npz", help="词表保存路径")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_files",  type=int, default=None, help="限制文件数（调试用）")
    args = parser.parse_args()

    segments = collect_segments(args.npz_dir, args.max_files)
    vocab = MotionVocabulary(vocab_size=args.vocab_size)
    vocab.fit(segments)
    vocab.save(args.save)
