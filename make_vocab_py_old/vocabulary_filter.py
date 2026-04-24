"""
vocabulary.py
=============
为 DiffusionPlanner 真实数据格式构建 512-token 运动词表。

真实 .npz 字段格式（已确认）
-----------------------------
  ego_agent_future        : (80, 3)      [x, y, heading]  ego局部坐标系, 8s@10Hz
  neighbor_agents_future  : (32, 80, 3)  同上，每个agent一行

过滤策略（v2 新增）
------------------
  1. 超速过滤：0.5s 内位移 > 20m（> 144 km/h）→ 传感器噪声
  2. 转向过滤：单 token 内 |dheading| > 90° → 物理上不可能

用法
----
  python vocabulary.py --npz_dir /path/to/cache --save ./vocab_512.npz --vocab_size 512
  from vocabulary import MotionVocabulary
  vocab = MotionVocabulary.load("vocab_512.npz")
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

MAX_SPEED_MS  = 40.0                  # m/s → 144 km/h
MAX_ANGLE_RAD = np.radians(90.0)      # rad


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
    """
    过滤噪声 segments。

    规则 1 — 超速过滤
        0.5s 内合位移 > 20m（等效速度 > 144 km/h）视为传感器噪声。
        分析旧词表时发现最高达到 548 km/h，这些数据会占用词表空间。

    规则 2 — 异常转向过滤
        单个 token（0.5s）内 |dheading| > 90°，
        真实车辆动力学不可能在 0.5s 内完成如此大的转向。
    """
    n_before = len(segments)

    speed = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2) / DT_TOKEN
    mask_speed = speed <= MAX_SPEED_MS
    mask_angle = np.abs(segments[:, 2]) <= MAX_ANGLE_RAD
    mask = mask_speed & mask_angle
    result = segments[mask]

    n_speed = (~mask_speed).sum()
    n_angle = (~mask_angle).sum()
    n_after = len(result)

    print(f"[Filter] 原始 segments         : {n_before:,}")
    print(f"[Filter] 超速异常 (>144 km/h)  : -{n_speed}")
    print(f"[Filter] 转向异常 (>90°/0.5s)  : -{n_angle}")
    print(f"[Filter] 过滤后保留            : {n_after:,}  ({n_after/n_before*100:.1f}%)")

    # 速度分布
    spd = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2) / DT_TOKEN
    bins   = [0, 1, 3, 6, 10, 20, MAX_SPEED_MS + 1]
    labels = ['静止  <1 m/s ', '低速 1-3 m/s', '慢速 3-6 m/s',
              '中速 6-10m/s', '快速10-20m/s', '高速20-40m/s']
    print("[Filter] 过滤后速度分布:")
    for i, label in enumerate(labels):
        cnt = int(((spd >= bins[i]) & (spd < bins[i+1])).sum())
        pct = cnt / n_after * 100 if n_after else 0
        bar = '█' * min(cnt * 40 // max(n_after, 1), 40)
        print(f"  {label}: {cnt:7,d}  ({pct:5.1f}%)  {bar}")

    return result


def stratified_sample(segments: np.ndarray, target_per_bin: int = 50_000,
                      seed: int = 42) -> np.ndarray:
    """
    按速度区间对 segments 做分层采样，确保词表聚类时各速度段均有充分代表。

    速度区间（单位：m/token，即 0.5s 内的位移）
    ──────────────────────────────────────────
      [0,   0.5)  静止/极低速
      [0.5, 1.0)  低速
      [1.0, 2.5)  慢速
      [2.5, 5.0)  中速
      [5.0,10.0)  快速
      [10,   ∞)   高速

    每个区间最多采样 target_per_bin 条，不足则全部保留。
    """
    rng = np.random.default_rng(seed)
    speed = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
    bins  = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, np.inf]
    labels = ['静止  <0.5m', '低速 0.5-1m', '慢速 1-2.5m',
              '中速 2.5-5m', '快速  5-10m', '高速  >10m ']

    sampled = []
    print("[Stratify] 分层采样统计:")
    for i, label in enumerate(labels):
        mask = (speed >= bins[i]) & (speed < bins[i + 1])
        idx  = np.where(mask)[0]
        n_avail  = len(idx)
        n_sample = min(n_avail, target_per_bin)
        chosen   = rng.choice(idx, size=n_sample, replace=False)
        sampled.append(segments[chosen])
        print(f"  {label}: 可用 {n_avail:8,d}  采样 {n_sample:7,d}")

    result = np.concatenate(sampled, axis=0)
    rng.shuffle(result)
    print(f"[Stratify] 采样后总量: {len(result):,}  (原始: {len(segments):,})")
    return result


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
                f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})，"
                "请增加数据量或降低 vocab_size")
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

    def encode(self, segments: np.ndarray) -> np.ndarray:
        self._check()
        raw = self._predictor.predict(self._scale(segments))
        return (raw + self.N_SPECIAL).astype(np.int64)

    def encode_topk(self, segments: np.ndarray, k=5) -> np.ndarray:
        self._check()
        Xs = self._scale(segments)
        Cs = self._scale(self._centroids)
        diff  = Xs[:, None] - Cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        topk  = np.argsort(dists, axis=-1)[:, :k]
        return (topk + self.N_SPECIAL).astype(np.int64)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        self._check()
        flat = np.asarray(indices, dtype=np.int64).ravel()
        out  = np.zeros((len(flat), 3), dtype=np.float32)
        ok   = flat >= self.N_SPECIAL
        if ok.any():
            raw = (flat[ok] - self.N_SPECIAL).clip(0, self.vocab_size - 1)
            out[ok] = self._centroids[raw]
        return out.reshape(indices.shape + (3,))

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

    @property
    def total_vocab_size(self):
        return self.vocab_size + self.N_SPECIAL

    @property
    def centroids(self):
        self._check()
        return self._centroids

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


# ── collect_segments ─────────────────────────────────────────────────────

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
            if len(s): all_segs.append(s)

        if "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]
            for n in range(nbr.shape[0]):
                if np.allclose(nbr[n, :TOKEN_STEP, :2], 0):
                    continue
                s = extract_segments(nbr[n])
                if len(s): all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未能从 {npz_dir} 收集到任何 segments")

    raw = np.concatenate(all_segs, axis=0)
    print(f"[Vocab] 收集完毕，开始过滤噪声 ...")
    return filter_segments(raw)


def collect_segments_stratified(npz_dir: str, max_files=None,
                                target_per_bin: int = 50_000) -> np.ndarray:
    """collect_segments + 按速度一维分层采样，用于构建平衡词表。"""
    segments = collect_segments(npz_dir, max_files)
    return stratified_sample(segments, target_per_bin=target_per_bin)


def stratified_sample_2d(segments: np.ndarray, target_per_bin: int = 15_000,
                         bin_targets: np.ndarray = None,
                         seed: int = 42) -> np.ndarray:
    """
    按速度×转向角二维分层采样，确保词表覆盖"高速弯道"等关键运动模式。

    速度区间（m/token，即 0.5s 内的位移量）
    ─────────────────────────────────────────
      [0,   0.5)  静止/极低速
      [0.5, 1.0)  低速
      [1.0, 2.5)  慢速
      [2.5, 5.0)  中速
      [5.0,10.0)  快速
      [10,    ∞)  高速

    转向角区间（|dθ|，单位度/0.5s）
    ──────────────────────────────
      [ 0°,  5°)  直行
      [ 5°, 15°)  微弯
      [15°, 45°)  中弯
      [45°, 90°)  急弯

    参数
    ----
    target_per_bin : int
        所有 bin 统一上限（bin_targets 为 None 时使用）
    bin_targets : np.ndarray, shape (6, 4), optional
        每个 bin 的独立采样上限，按驾驶场景常见程度设定。
        优先级高于 target_per_bin。
    """
    # 默认按常见程度设置的目标矩阵（行=速度，列=转向角）
    # 常见的低中速直行/微弯给更多名额，罕见的急弯保留自然数量
    DEFAULT_BIN_TARGETS = np.array([
        # 直行<5°  微弯5-15°  中弯15-45°  急弯45-90°
        [20_000,   10_000,     5_000,      1_000],   # 静止
        [25_000,   20_000,     8_000,      1_000],   # 低速
        [20_000,   15_000,     5_000,        500],   # 慢速
        [25_000,   20_000,     3_000,        500],   # 中速
        [20_000,    5_000,     3_000,      4_000],   # 快速
        [15_000,    9_000,     8_000,     10_000],   # 高速
    ], dtype=np.int64)

    rng = np.random.default_rng(seed)
    speed = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
    angle = np.abs(segments[:, 2])  # radians

    spd_bins   = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, np.inf]
    spd_labels = ['静止 <0.5m', '低速0.5-1m', '慢速1-2.5m',
                  '中速2.5-5m', '快速5-10m ', '高速 >10m ']

    ang_bins   = [0.0, np.radians(5), np.radians(15), np.radians(45), np.inf]
    ang_labels = ['直行<5° ', '微弯5-15°', '中弯15-45°', '急弯45-90°']

    targets = bin_targets if bin_targets is not None else DEFAULT_BIN_TARGETS

    sampled = []
    header = f"  {'':12s}" + "".join(f"  {al:>11s}" for al in ang_labels)
    print("[Stratify2D] 二维分层采样统计 (速度 × 转向角)，格式: 可用/采样")
    print(header)

    for i, sl in enumerate(spd_labels):
        spd_mask = (speed >= spd_bins[i]) & (speed < spd_bins[i + 1])
        row_parts = []
        for j in range(len(ang_labels)):
            ang_mask = (angle >= ang_bins[j]) & (angle < ang_bins[j + 1])
            idx = np.where(spd_mask & ang_mask)[0]
            n_avail  = len(idx)
            n_sample = min(n_avail, int(targets[i, j]))
            if n_sample > 0:
                chosen = rng.choice(idx, size=n_sample, replace=False)
                sampled.append(segments[chosen])
            row_parts.append(f"{n_avail:>5,}/{n_sample:<5,}")
        print(f"  {sl:12s}" + "".join(f"  {p:>11s}" for p in row_parts))

    result = np.concatenate(sampled, axis=0)
    rng.shuffle(result)
    print(f"[Stratify2D] 采样后总量: {len(result):,}  (原始: {len(segments):,})")
    return result


def collect_segments_stratified_2d(npz_dir: str, max_files=None,
                                   target_per_bin: int = 15_000) -> np.ndarray:
    """collect_segments + 速度×转向角二维分层采样（按自然频率加权），用于构建路径感知词表。"""
    segments = collect_segments(npz_dir, max_files)
    return stratified_sample_2d(segments, target_per_bin=target_per_bin)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir",        required=True)
    parser.add_argument("--save",           default="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/npz2token_dataset/vocab_512_filter.npz")
    parser.add_argument("--vocab_size",     type=int,   default=512)
    parser.add_argument("--max_files",      type=int,   default=None)
    parser.add_argument("--stratify",       action="store_true",
                        help="按速度一维分层采样")
    parser.add_argument("--stratify_2d",    action="store_true",
                        help="按速度×转向角二维分层采样（推荐，覆盖高速弯道等关键模式）")
    parser.add_argument("--target_per_bin", type=int,   default=15_000,
                        help="每个 bin 的最大样本数（2D 模式默认 15000，1D 模式建议 50000）")
    args = parser.parse_args()

    if args.stratify_2d:
        segments = collect_segments_stratified_2d(
            args.npz_dir, args.max_files, args.target_per_bin)
    elif args.stratify:
        segments = collect_segments_stratified(
            args.npz_dir, args.max_files, args.target_per_bin)
    else:
        segments = collect_segments(args.npz_dir, args.max_files)

    vocab = MotionVocabulary(vocab_size=args.vocab_size)
    vocab.fit(segments)
    vocab.save(args.save)
