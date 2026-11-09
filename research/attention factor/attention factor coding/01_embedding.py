"""
环节 ①：特征嵌入（Attention Factors for Statistical Arbitrage 复现）

原文：[ATT] 3.2 节 "Attention Factors"
    X̃_t = X_t W_K,   W_K ∈ R^{M×d},  M = 79,  d = 32（附录 B Table 4: d_x = 32）

这个脚本做两件事：
  Part 1  把 data/features.parquet（长表）整理成模型直接能用的"按天堆叠"的数组：
          每天 500 个位置（当天股票不足 500 只时补空位，用 mask 标记）
  Part 2  演示嵌入层：用第一个训练窗口（1990-1997）拟合水平量标准化，
          对 1998 年第一个交易日的 500 只股票做 X̃ = X W_K，检查形状、数值和梯度

输入：
    data/features.parquet

输出（都在 attention factor/ 下）：
    X.npy       (D, 500, 79) float32  每天每只股票的 79 维特征 X_{t-1}（未做水平量标准化）
    R.npy       (D, 500)     float32  t 日收益 R_t（缺失记 0）
    mask.npy    (D, 500)     bool     该位置是否有股票
    permno.npy  (D, 500)     int64    该位置的股票代码（空位为 0）
    dates.npy   (D,)         datetime64[D]  交易日 t
    feature_cols.json                 79 列的列名和顺序，以及特征分组
    01_embedding.log                  说明与运行结果

    说明：水平量（中位数、rf）的标准化依赖训练窗口，所以 X.npy 存的是未标准化的值，
    每个滚动窗口训练时再用 af_model.LevelNormalizer 拟合、转换。

用法（在 research 目录下）：
    python "attention factor/attention factor coding/01_embedding.py"
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))     # research/attention factor/attention factor coding
AF_DIR = os.path.dirname(HERE)                        # research/attention factor
ROOT = os.path.dirname(AF_DIR)                        # research
sys.path.insert(0, HERE)
from af_model import N_CHARS, N_FEAT, FeatureEmbedding, LevelNormalizer, load_feature_cols  # noqa: E402

FEAT_FILE = os.path.join(ROOT, "data", "features.parquet")
LOG_PATH = os.path.join(AF_DIR, "01_embedding.log")

START = "1990-01-01"          # [ATT] 4.1 节：数据从 1990 年 1 月开始
N_SLOTS = 500                 # 每天最多 500 只股票
D_EMB = 32                    # [ATT] 附录 B Table 4: d_x = 32
FIRST_TRAIN = ("1990-01-01", "1997-12-31")   # 第一个 8 年训练窗口（[ATT] 4.2 节）
DEMO_DAY = "1998-01-02"                      # 样本外第一天
SEED = 0

t0 = time.time()
L = []


def log(msg=""):
    print(msg)
    L.append(str(msg))


# ===========================================================================
# Part 1  长表 -> 按天堆叠的数组
# ===========================================================================
feat_cols = load_feature_cols(FEAT_FILE)
groups = json.loads(pq.read_schema(FEAT_FILE).metadata[b"feature_groups"])
assert len(feat_cols) == N_FEAT

df = pd.read_parquet(FEAT_FILE, columns=["DATE", "PERMNO", "RET_ADJ"] + feat_cols,
                     filters=[("DATE", ">=", pd.Timestamp(START)), ("in_universe", "==", 1)])
df = df.sort_values(["DATE", "PERMNO"]).reset_index(drop=True)

dates, day_idx = np.unique(df["DATE"].values, return_inverse=True)
slot = df.groupby("DATE").cumcount().values
D = len(dates)
assert slot.max() < N_SLOTS

X = np.zeros((D, N_SLOTS, N_FEAT), dtype="float32")
R = np.zeros((D, N_SLOTS), dtype="float32")
mask = np.zeros((D, N_SLOTS), dtype=bool)
permno = np.zeros((D, N_SLOTS), dtype="int64")

X[day_idx, slot] = df[feat_cols].values.astype("float32")
ret = df["RET_ADJ"].values
n_ret_nan = int(np.isnan(ret).sum())
R[day_idx, slot] = np.nan_to_num(ret, nan=0.0)
mask[day_idx, slot] = True
permno[day_idx, slot] = df["PERMNO"].values
n_rows = len(df)
del df

np.save(os.path.join(AF_DIR, "X.npy"), X)
np.save(os.path.join(AF_DIR, "R.npy"), R)
np.save(os.path.join(AF_DIR, "mask.npy"), mask)
np.save(os.path.join(AF_DIR, "permno.npy"), permno)
np.save(os.path.join(AF_DIR, "dates.npy"), dates.astype("datetime64[D]"))
with open(os.path.join(AF_DIR, "feature_cols.json"), "w") as fh:
    json.dump({"feature_cols": feat_cols, "feature_groups": groups}, fh, indent=1, ensure_ascii=False)

n_per_day = mask.sum(1)
dts = pd.DatetimeIndex(dates)

# ===========================================================================
# Part 2  嵌入层演示
# ===========================================================================
tr = (dts >= FIRST_TRAIN[0]) & (dts <= FIRST_TRAIN[1])
norm = LevelNormalizer(feat_cols).fit(X[tr])

d0 = int(np.flatnonzero(dts >= DEMO_DAY)[0])
Xd = norm.transform(X[d0])                       # (500, 79)
md = mask[d0]
level_before = X[tr][:, 0, N_CHARS:]
level_after = norm.transform(X[tr][:, :1, :])[:, 0, N_CHARS:]

torch.manual_seed(SEED)
emb = FeatureEmbedding(n_features=N_FEAT, d=D_EMB)
x_t = torch.from_numpy(Xd)
x_tilde = emb(x_t)                               # (500, 32)

# 批量（多天一起）也能直接算：(B, 500, 79) -> (B, 500, 32)
xb = torch.from_numpy(norm.transform(X[d0:d0 + 5]))
xb_tilde = emb(xb)

# 梯度检查：随便构造一个标量，反向传播，W_K 应该拿到梯度
loss = (x_tilde[torch.from_numpy(md)] ** 2).mean()
loss.backward()
grad_norm = emb.W_K.weight.grad.norm().item()

# 手算一遍 X @ W_K，确认与模块输出一致
W_K = emb.W_K.weight.detach().numpy().T          # (79, 32)
max_diff = np.abs(Xd @ W_K - x_tilde.detach().numpy()).max()

# ===========================================================================
# 写 log
# ===========================================================================
log("=" * 78)
log("环节 ①：特征嵌入")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.2 节 \"Attention Factors\": \"for each point in time, the firm characteristics")
log("        X_t are first embedded as X̃_t = X_t W_K, W_K ∈ R^{M×d}\"")
log("  [ATT] 4.1 节: X_{t-1} 的维度 M = 79（39 个特征排序值 + 39 个截面中位数 + 无风险利率）")
log("  [ATT] 附录 B Table 4: \"d_x = 32, Model attention dim\"，即 d = 32")
log("")
log("二、本环节的实现（代码见 af_model.py）")
log("  1. FeatureEmbedding：不带偏置的线性层 nn.Linear(79, 32, bias=False)，")
log("     forward 计算 X̃ = X @ W_K。W_K 是可学习参数，和后面的环节一起训练。")
log("     初始化：PyTorch 默认（Kaiming uniform）。原文未说明，自行设定。")
log("  2. LevelNormalizer：对 40 个水平量列（39 个中位数 + rf）做标准化，39 个排序值列不动。")
log("     - 先对量级跨越多个数量级的 8 列取对数：" + ", ".join(
    c.replace("med_", "") for c in ["med_AT", "med_LME", "med_Vol", "med_Variance",
                                     "med_Resid_Var", "med_Spread", "med_STD_W1", "med_LTurnover"]))
log("     - 再用训练窗口内的均值、标准差做 z-score；样本外只用训练窗口的统计量。")
log("     - 原因：排序值在 [-0.5, 0.5]，而中位数是原始量纲（如 med_AT 约 1 万），")
log("       不处理的话大数值的列会主导嵌入结果。")
log("     - 原文未说明这一步，自行设定。")
log("")
log("三、输出的数组（attention factor/ 下）")
log(f"  X.npy       {X.shape}  float32  特征 X_{{t-1}}（水平量未标准化）")
log(f"  R.npy       {R.shape}      float32  t 日收益 R_t")
log(f"  mask.npy    {mask.shape}      bool     该位置是否有股票")
log(f"  permno.npy  {permno.shape}      int64    股票代码，空位为 0")
log(f"  dates.npy   ({D},)           日期 {dts[0].date()} -> {dts[-1].date()}")
log(f"  feature_cols.json            79 列列名顺序 + 特征分组")
log(f"  每天的股票按 PERMNO 从小到大排在前面，不足 500 的位置补 0，mask = False。")
log(f"  共 {n_rows:,} 个 (股票, 日) 观测；每天股票数 {n_per_day.min()}-{n_per_day.max()}，"
    f"平均 {n_per_day.mean():.1f}")
log(f"  股票池内 t 日收益缺失 {n_ret_nan} 个，记为 0（这些股票当天收益为 0，不影响形状）")
log("")
log("四、演示：1998 年第一个交易日（样本外第一天）")
log(f"  标准化用第一个训练窗口 {FIRST_TRAIN[0]} ~ {FIRST_TRAIN[1]}（{tr.sum()} 天）拟合")
log(f"  水平量 40 列标准化前：每列均值的范围 [{level_before.mean(0).min():.3g}, "
    f"{level_before.mean(0).max():.3g}]")
log(f"  水平量 40 列标准化后（训练窗口内）：均值 ≈ {np.abs(level_after.mean(0)).max():.1e}，"
    f"标准差 ≈ {level_after.std(0).mean():.3f}")
log(f"  演示日 {dts[d0].date()}：{md.sum()} 只股票")
log(f"  输入 X:     {tuple(x_t.shape)}")
log(f"  权重 W_K:   {tuple(W_K.shape)}（可学习参数 {W_K.size:,} 个）")
log(f"  输出 X̃:     {tuple(x_tilde.shape)}")
log(f"  批量输入 {tuple(xb.shape)} -> 输出 {tuple(xb_tilde.shape)}")
log(f"  X @ W_K 手算与模块输出的最大差: {max_diff:.2e}（应接近 0）")
log(f"  反向传播后 W_K 梯度范数: {grad_norm:.4f}（> 0 说明可以训练）")
xt = x_tilde.detach().numpy()[md]
log(f"  X̃ 数值（随机初始化，未训练）：均值 {xt.mean():.4f}，标准差 {xt.std():.4f}")
log("  前 3 只股票、前 5 个嵌入维度：")
log("    PERMNO   " + "  ".join(f"dim{j:<5d}" for j in range(5)))
for i in range(3):
    log(f"    {permno[d0, i]:<8d} " + "  ".join(f"{v:+.4f}  " for v in xt[i, :5]))
log("")
log("五、下一环节")
log("  环节 ②：注意力因子权重 ω^F = Softmax(Q X̃ᵀ / √d)，Q ∈ R^{K×d}")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(L) + "\n")
