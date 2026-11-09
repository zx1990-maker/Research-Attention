"""
环节 ④：过去 30 天的残差 —— 数据整理 + 检查（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 3.3 节 "Arbitrage Portfolio"：ω^port_{i,t-1} = LongConv_θ(ε_{i,(t-s,t-1)})
  [ATT] 4.2 节："All sequence models use a look-back window of the past 30 daily residual returns"，s = 30
  [19]  Guijarro-Ordonez, Pelger, Zanotti, Deep Learning Statistical Arbitrage（arXiv:2106.04028）：
        公式 (1) ε_t = Φ_{t-1} R_t；III.B 节 "daily residuals out of sample ... the residual composition
        matrix Φ_{t-1} ... depends only on the information up to time t-1"；
        II.B 节 ε^L_{n,t-1} := (ε_{n,t-L} ··· ε_{n,t-1})；II.D 节输入为累计残差 x := Int(ε^L_{n,t-1})

实现的做法（与 [19] 一致）：
  1. 每天用当天自己的 ω^ε 算当天的残差：ε_s = ω^ε_{s-1} R_s（做法 B）
  2. 交易日 t 取 ε_{t-30} ... ε_{t-1}，累加成"残差价格"路径，作为 LongConv 的输入
  3. 截面：第 m 个月的每个交易日 t，用当月股票池的 500 只股票；它们过去 30 天的残差也在
     这 500 只股票组成的截面上计算（用各自当天的特征）。月初新进股票池的股票，进池前的特征
     由 data/coding/02 额外输出（in_universe = 0 的行）。

这个脚本做两件事：
  Part 1  数据整理：把 features.parquet 整理成"按月的块"，每块 = 当月股票池 500 只 ×
          （往前 30 个交易日 + 当月所有交易日）
  Part 2  检查：截面是否正确、30 天历史覆盖率、残差与手算一致、不使用未来信息、
          累计正确、梯度可传、计算耗时

输入：data/features.parquet
输出（attention factor/ 下）：
    hist_X.npy        (n_rows+1, 79) float32  所有需要的 (股票, 日) 的特征，第 0 行是全 0 的空位
    hist_R.npy        (n_rows+1,)    float32  对应的当日收益（缺失记 0）
    hist_idx.npy      (Σ块天数, 500) int32    每块每天每个位置对应 hist_X 的行号，0 = 无数据
    hist_day.npy      (Σ块天数,)     int32    对应 hist_dates 的日期位置
    hist_ptr.npy      (块数+1,)      int64    第 b 块在 hist_idx 中的起止行
    hist_permno.npy   (块数, 500)    int64    第 b 块（第 b 个月）的股票，空位为 0
    hist_months.json                          第 b 块对应的月份
    hist_dates.npy                            日期
    04_residual_history.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/04_residual_history.py"
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AF_DIR)
sys.path.insert(0, HERE)
from af_model import (LOOKBACK, N_FEAT, AttentionFactorWeights, FeatureEmbedding,  # noqa: E402
                      LevelNormalizer, ResidualProjection, daily_residuals, load_feature_cols,
                      residual_windows)

FEAT_FILE = os.path.join(ROOT, "data", "features.parquet")
LOG_PATH = os.path.join(AF_DIR, "04_residual_history.log")
LOAD_FROM = "1989-10-01"        # 1990 年 1 月的回看窗口需要 1989 年 11-12 月的数据
FIRST_MONTH = "1990-01"
N_SLOTS = 500
L = LOOKBACK
K_CHECK = 8
D_EMB = 32
SEED = 0
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
DEMO_MONTH = "1998-01"

t0 = time.time()
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def tick():
    return f"[{time.time() - t0:.0f}s]"


# ===========================================================================
# Part 1  数据整理
# ===========================================================================
print("Part 1  数据整理")
feat_cols = load_feature_cols(FEAT_FILE)
df = pd.read_parquet(FEAT_FILE, columns=["DATE", "PERMNO", "in_universe", "RET_ADJ"] + feat_cols,
                     filters=[("DATE", ">=", pd.Timestamp(LOAD_FROM))])
df = df.sort_values(["DATE", "PERMNO"]).reset_index(drop=True)
n_rows = len(df)
n_pre_rows = int((df["in_universe"] == 0).sum())

dates, day = np.unique(df["DATE"].values, return_inverse=True)
perms, pcol = np.unique(df["PERMNO"].values, return_inverse=True)
D_all, P = len(dates), len(perms)

hist_X = np.zeros((n_rows + 1, N_FEAT), dtype="float32")
hist_X[1:] = df[feat_cols].values.astype("float32")
hist_R = np.zeros(n_rows + 1, dtype="float32")
hist_R[1:] = np.nan_to_num(df["RET_ADJ"].values.astype("float32"), nan=0.0)
grid = np.zeros((D_all, P), dtype="int32")            # (日, 股票) -> hist_X 行号
grid[day, pcol] = np.arange(1, n_rows + 1, dtype="int32")
isU = np.zeros((D_all, P), dtype=bool)
isU[day, pcol] = df["in_universe"].values == 1
del df

per = pd.DatetimeIndex(dates).to_period("M")
mcode = pd.factorize(per)[0]
mstart = np.flatnonzero(np.r_[True, mcode[1:] != mcode[:-1]])
mend = np.r_[mstart[1:] - 1, D_all - 1]
months_all = per[mstart]

blk_idx, blk_day, blk_ptr, blk_permno, blk_months = [], [], [0], [], []
for m, mo in enumerate(months_all):
    if mo < pd.Period(FIRST_MONTH):
        continue
    cols = np.flatnonzero(isU[mstart[m]:mend[m] + 1].any(axis=0))   # 当月股票池
    assert len(cols) <= N_SLOTS
    d0 = mstart[m] - L
    assert d0 >= 0
    days = np.arange(d0, mend[m] + 1)
    idx = np.zeros((len(days), N_SLOTS), dtype="int32")
    idx[:, :len(cols)] = grid[days][:, cols]
    pm = np.zeros(N_SLOTS, dtype="int64")
    pm[:len(cols)] = perms[cols]
    blk_idx.append(idx)
    blk_day.append(days.astype("int32"))
    blk_ptr.append(blk_ptr[-1] + len(days))
    blk_permno.append(pm)
    blk_months.append(str(mo))
blk_idx = np.concatenate(blk_idx)
blk_day = np.concatenate(blk_day)
blk_ptr = np.array(blk_ptr, dtype="int64")
blk_permno = np.stack(blk_permno)
B = len(blk_months)

np.save(os.path.join(AF_DIR, "hist_X.npy"), hist_X)
np.save(os.path.join(AF_DIR, "hist_R.npy"), hist_R)
np.save(os.path.join(AF_DIR, "hist_idx.npy"), blk_idx)
np.save(os.path.join(AF_DIR, "hist_day.npy"), blk_day)
np.save(os.path.join(AF_DIR, "hist_ptr.npy"), blk_ptr)
np.save(os.path.join(AF_DIR, "hist_permno.npy"), blk_permno)
np.save(os.path.join(AF_DIR, "hist_dates.npy"), dates.astype("datetime64[D]"))
with open(os.path.join(AF_DIR, "hist_months.json"), "w") as fh:
    json.dump(blk_months, fh)
print(f"  {B} 个月块，共 {len(blk_idx):,} 个块-日 {tick()}")


def block(b):
    """第 b 块：idx (nd, 500)，日期位置 (nd,)，该块交易日从第 L 天开始"""
    return blk_idx[blk_ptr[b]:blk_ptr[b + 1]], blk_day[blk_ptr[b]:blk_ptr[b + 1]]


# ===========================================================================
# Part 2  检查
# ===========================================================================
print("Part 2  检查")
dts = pd.DatetimeIndex(dates)
checks = []


def add(item, ok, val):
    checks.append((item, bool(ok), val))


# ---- C1 截面：块内交易日的有效股票 = 当天股票池 ---------------------------------
mismatch = 0
n_trade_days = 0
for b in range(B):
    idx, dd = block(b)
    cols = np.searchsorted(perms, blk_permno[b][blk_permno[b] > 0])
    tr_idx, tr_days = idx[L:], dd[L:]
    n_trade_days += len(tr_days)
    in_block = tr_idx[:, :len(cols)] > 0
    u_block = isU[tr_days][:, cols]
    mismatch += int((in_block != u_block).any(axis=1).sum())
    mismatch += int((isU[tr_days].sum(1) != u_block.sum(1)).sum())   # 当天股票池里没有块外的股票
add("C1 块内交易日的股票 = 当天股票池", mismatch == 0,
    f"{n_trade_days:,} 个交易日，不一致 {mismatch} 天")

# ---- C2 30 天历史覆盖率 -----------------------------------------------------
full_new = full_old = tot = 0
incomplete_perm = set()
reason_unlisted = 0
for b in range(B):
    idx, dd = block(b)
    cols = np.searchsorted(perms, blk_permno[b][blk_permno[b] > 0])
    nb = len(cols)
    v_new = idx[:, :nb] > 0                                   # 有特征和收益（含进池前的行）
    v_old = v_new & isU[dd][:, cols]                          # 只用"当天在股票池"的行
    for t in range(L, len(dd)):
        trade = v_new[t]                                      # 当天在股票池
        ok_new = v_new[t - L:t].all(0) & trade
        ok_old = v_old[t - L:t].all(0) & trade
        tot += int(trade.sum())
        full_new += int(ok_new.sum())
        full_old += int(ok_old.sum())
        bad = trade & ~ok_new
        if bad.any():
            for c in cols[bad]:
                incomplete_perm.add(int(perms[c]))
            # 窗口内缺的那几天，该股票在日度面板中是否完全没有记录（未上市/停牌）
            reason_unlisted += int((grid[dd[t - L:t]][:, cols[bad]] == 0).any(0).sum())
add("C2 30 天残差历史覆盖率提高", full_new > full_old,
    f"只用池内的行: {full_old / tot:.2%}；加上进池前的行: {full_new / tot:.2%}")

# ---- 模型（随机初始化），检查用 float64 ----------------------------------------
X01 = np.load(os.path.join(AF_DIR, "X.npy"), mmap_mode="r")
d01 = pd.DatetimeIndex(np.load(os.path.join(AF_DIR, "dates.npy")))
tr = np.flatnonzero((d01 >= FIRST_TRAIN[0]) & (d01 <= FIRST_TRAIN[1]))
norm = LevelNormalizer(feat_cols).fit(np.asarray(X01[tr[0]:tr[-1] + 1]))

torch.manual_seed(SEED)
emb = FeatureEmbedding(N_FEAT, D_EMB).double()
att = AttentionFactorWeights(K_CHECK, D_EMB).double()
proj = ResidualProjection()


def load(b, dtype=torch.float64):
    idx, dd = block(b)
    X = torch.from_numpy(norm.transform(hist_X[idx])).to(dtype)
    R = torch.from_numpy(hist_R[idx]).to(dtype)
    V = torch.from_numpy(idx > 0)
    return X, R, V, dd


b_demo = blk_months.index(DEMO_MONTH)
Xb, Rb, Vb, dd = load(b_demo)
with torch.no_grad():
    eps, w_eps = daily_residuals(emb, att, proj, Xb, Rb, Vb)
    x, ok = residual_windows(eps, Vb)

# ---- C3 残差与手算一致 ---------------------------------------------------------
s = L + 3                                     # 块内某一天
Wk = emb.W_K.weight.detach().numpy().T
Q = att.Q.detach().numpy()
Xs, Rs, vs = Xb[s].numpy(), Rb[s].numpy(), Vb[s].numpy()
sc = Q @ (Xs @ Wk).T / np.sqrt(D_EMB)
sc = np.where(vs[None, :], sc, -np.inf)
wF = np.exp(sc - sc.max(1, keepdims=True))
wF /= wF.sum(1, keepdims=True)
bT = wF.T @ np.linalg.inv(wF @ wF.T + proj.ridge * np.eye(K_CHECK))
eps_np = ((np.eye(N_SLOTS) - bT @ wF) * np.outer(vs, vs)) @ Rs
d3 = np.abs(eps_np - eps[s].numpy()).max()
add("C3 每日残差 = 手算 ω^ε_{s-1} R_s", d3 < 1e-10, f"{dts[dd[s]].date()}，最大差 {d3:.1e}")

# 批量（整块一次算）与逐天算一致
with torch.no_grad():
    eps_loop = torch.stack([daily_residuals(emb, att, proj, Xb[i:i + 1], Rb[i:i + 1], Vb[i:i + 1])[0][0]
                            for i in range(len(dd))])
d3b = (eps_loop - eps).abs().max().item()
add("C3 整块批量 = 逐天计算", d3b < 1e-10, f"最大差 {d3b:.1e}")

# ---- C4 不使用未来信息 ---------------------------------------------------------
t_pos = L + 5                                 # 块内交易日 t
i_win = t_pos - L                             # 对应窗口编号
R2 = Rb.clone()
R2[t_pos] += 0.05                             # 改动 t 日收益
X2 = Xb.clone()
X2[t_pos] += 1.0                              # 改动 t 日那一行的特征（X_{t-1}，交易日 t 要用，但不进窗口）
with torch.no_grad():
    eps2, _ = daily_residuals(emb, att, proj, X2, R2, Vb)
    x2, _ = residual_windows(eps2, Vb)
same_t = (x2[i_win] - x[i_win]).abs().max().item()
diff_t1 = (x2[i_win + 1] - x[i_win + 1]).abs().max().item()
add("C4 改动 t 日数据，t 日的窗口不变", same_t == 0, f"t 日窗口变化 {same_t:.1e}")
add("C4 改动 t 日数据，t+1 日的窗口改变", diff_t1 > 0, f"t+1 日窗口变化 {diff_t1:.1e}")

# ---- C5 累计正确 ---------------------------------------------------------------
c_last = (x[i_win, :, -1] - eps[t_pos - L:t_pos].sum(0)).abs().max().item()
c_first = (x[i_win, :, 0] - eps[t_pos - L]).abs().max().item()
add("C5 累计残差：第 1 个 = ε_{t-30}，最后 1 个 = 30 天之和", max(c_last, c_first) < 1e-12,
    f"最大差 {max(c_last, c_first):.1e}")
add("C5 窗口形状 = (交易日数, 500, 30)", tuple(x.shape) == (len(dd) - L, N_SLOTS, L), str(tuple(x.shape)))

# ---- C6 梯度 -------------------------------------------------------------------
torch.manual_seed(SEED)
emb32, att32 = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB)
X32, R32, V32, _ = load(b_demo, torch.float32)
e32, _ = daily_residuals(emb32, att32, proj, X32, R32, V32)
x32, ok32 = residual_windows(e32, V32)
(x32[ok32] ** 2).mean().backward()
gq, gw = att32.Q.grad.norm().item(), emb32.W_K.weight.grad.norm().item()
add("C6 梯度从窗口传回 Q 和 W_K", gq > 0 and gw > 0, f"|∇Q| {gq:.2e}，|∇W_K| {gw:.2e}")

# ---- C7 同一天在相邻两块中的残差（截面不同造成的差异，仅供参考）--------------------
Xa, Ra, Va, da = load(b_demo + 1)
with torch.no_grad():
    eps_a, _ = daily_residuals(emb, att, proj, Xa, Ra, Va)
common_days = np.intersect1d(dd[L:], da[:L])
pm0, pm1 = blk_permno[b_demo], blk_permno[b_demo + 1]
common_p = np.intersect1d(pm0[pm0 > 0], pm1[pm1 > 0])
i0 = np.searchsorted(dd, common_days)
i1 = np.searchsorted(da, common_days)
j0 = np.flatnonzero(np.isin(pm0, common_p))
j1 = np.flatnonzero(np.isin(pm1, common_p))
assert (pm0[j0] == pm1[j1]).all()
e0 = eps[i0][:, j0].numpy()
e1 = eps_a[i1][:, j1].numpy()
c7_rel = np.abs(e0 - e1).mean() / np.abs(e0).mean()
c7_corr = np.corrcoef(e0.ravel(), e1.ravel())[0, 1]

# ---- 计算耗时 ------------------------------------------------------------------
timing = {}
for K in [8, 30, 100]:
    torch.manual_seed(SEED)
    e_, a_ = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K, D_EMB)
    X32, R32, V32, _ = load(b_demo, torch.float32)
    t1 = time.time()
    for _ in range(3):
        ee, _ = daily_residuals(e_, a_, proj, X32, R32, V32)
        xx, oo = residual_windows(ee, V32)
        (xx[oo] ** 2).mean().backward()
    timing[K] = (time.time() - t1) / 3

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(c[1] for c in checks)
log("=" * 78)
log("环节 ④：过去 30 天的残差 —— 数据整理与检查")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.3 节 \"Arbitrage Portfolio\": \"The arbitrage portfolio weight function depends on the")
log("        time-series signals that we extract with LongConv from the past s residuals ε_{i,(t-s,t-1)}\"")
log("  [ATT] 4.2 节: \"All sequence models use a look-back window of the past 30 daily residual returns.\"")
log("  [ATT] 没有说明这 30 天的残差用哪一天的 ω^ε 计算，也没有说明输入是每日残差还是累计残差。")
log("        [ATT] 4.2 节说 PCA 基准是 \"a benchmark in the spirit of [19]\"，交易成本 \"as in [19]\"，")
log("        因此这两处按 [19] 处理：")
log("  [19] Guijarro-Ordonez, Pelger, Zanotti, Deep Learning Statistical Arbitrage, arXiv:2106.04028")
log("        II.A 节公式 (1): ε_t = R_t - β_{t-1}ᵀ F_t = Φ_{t-1} R_t")
log("        III.B 节: \"we conduct a rolling window estimation to obtain daily residuals out of sample")
log("        ... This means that the residual composition matrix Φ_{t-1} of equation (1) depends only on")
log("        the information up to time t-1\"")
log("        II.B 节: \"We apply the signal extraction to the time-series of the last L lagged residuals")
log("        ε^L_{n,t-1} := (ε_{n,t-L} ··· ε_{n,t-1})\"")
log("        II.D 节: \"The input to the signal extraction functions are the last L cumulative residuals")
log("        ... x := Int(ε^L_{n,t-1}) ... the operation Int(·) simply integrates a discrete time-series.")
log("        We can view the cumulative residuals as the residual 'price' process.\"")
log("        III.B 节: \"At each day we only consider the stocks with no missing observations in the")
log("        daily returns within the rolling window\"")
log("")
log("二、实现")
log("  1. 每日残差（做法 B）：第 s 天的残差用第 s 天自己的投影 ε_s = ω^ε_{s-1} R_s，")
log("     ω^ε_{s-1} 由第 s 天那一行的特征 X_{s-1} 经环节 ①②③ 得到。见 af_model.daily_residuals。")
log("     依据 [19] 公式 (1) 与 \"daily residuals out of sample\"。")
log("  2. 回看窗口：交易日 t 取 ε_{t-30}, ..., ε_{t-1}（不含 t 日），累加成残差价格路径")
log("     (ε_{t-30}, ε_{t-30}+ε_{t-29}, ..., Σ 30 天)，作为 LongConv 的输入。见 af_model.residual_windows。")
log("     依据 [19] II.B、II.D 节（Int 累计）。")
log("  3. 截面（自行设定）：第 m 月每个交易日用当月股票池（固定的 500 只）；这 500 只股票过去")
log("     30 天的残差也在这 500 只组成的截面上、用它们各自当天的特征计算。")
log("     月初新进股票池的股票，进池前那些天的特征来自 data/coding/02 额外输出的 in_universe = 0 的行")
log("     （排序值相对于当天股票池的分布计算）。")
log("     原因：做法 B 需要每只股票在过去每一天的特征；如果只用当天在池子里的股票，")
log("     新进股票就没有残差历史。")
log("  4. 缺失（依据 [19]）：窗口内有任何一天没有数据（未上市、停牌）的股票，当天不交易")
log("     （residual_windows 返回的 ok = False），但仍然参与构造因子。")
log("  5. 训练时按月组织：每个月一块，块内 = 往前 30 个交易日 + 当月所有交易日，")
log("     每天的残差只算一次，被后面 30 个交易日的窗口共用。")
log("")
log("三、输出（attention factor/ 下）")
log(f"  hist_X.npy       ({n_rows + 1:,}, 79)  所有需要的 (股票, 日) 的特征（第 0 行为空位）")
log(f"                   其中股票池内 {n_rows - n_pre_rows:,} 行，进池前 {n_pre_rows:,} 行")
log(f"  hist_R.npy       ({n_rows + 1:,},)      对应的当日收益")
log(f"  hist_idx.npy     {blk_idx.shape}  每块每天每个位置在 hist_X 中的行号，0 = 无数据")
log(f"  hist_day.npy     {blk_day.shape}  对应 hist_dates 的位置")
log(f"  hist_ptr.npy     ({B + 1},)  第 b 块的起止")
log(f"  hist_permno.npy  {blk_permno.shape}  第 b 块的股票")
log(f"  hist_months.json  {B} 个月：{blk_months[0]} -> {blk_months[-1]}")
log(f"  每块 = 往前 {L} 个交易日 + 当月交易日，块内第 {L} 天起是当月交易日。")
log("")
log(f"四、检查结果：{n_pass} / {len(checks)} 项通过")
log(f"  （模型 K = {K_CHECK}，W_K、Q 随机初始化，λ_ridge = {proj.ridge:g}，演示月份 {DEMO_MONTH}）")
log("")
for item, ok_, val in checks:
    log(f"  [{'通过' if ok_ else '失败'}] {item}")
    log(f"         {val}")
log("")
log("  C2 补充：")
log(f"    仍不完整的 {tot - full_new:,} 个 (股票, 交易日)（占 {(tot - full_new) / tot:.2%}），"
    f"涉及 {len(incomplete_perm)} 只股票；")
log(f"    其中 {reason_unlisted:,} 个是窗口内该股票在日度面板中没有记录（刚上市不满 30 天或停牌），")
log("    这些按 [19] 的规则当天不交易。")
log("")
log("  C7 同一天的残差在相邻两块中是否一致（仅供参考，不算检查项）：")
log(f"    {DEMO_MONTH} 的交易日同时出现在 {DEMO_MONTH} 块（当月）和下一块（回看窗口）中。两块的截面")
log(f"    只差月初进出股票池的少数股票，因子权重会略有不同，所以残差不完全相同。")
log(f"    共同的 {len(common_days)} 天 × {len(common_p)} 只股票：平均绝对差 / 平均绝对残差 = {c7_rel:.2%}，"
    f"相关系数 {c7_corr:.4f}")
log("")
log("五、计算耗时（一个月块，float32，CPU，前向 + 反向）")
for K, sec in timing.items():
    log(f"  K = {K:>3}: {sec:.2f} 秒 / 月块；一个 8 年训练窗口（96 个月块）约 {sec * 96:.0f} 秒 / epoch")
log("  （尚未包含 LongConv；Table 4 中 epochs = 30）")
log("")
log("六、下一环节")
log("  环节 ⑤：LongConv 交易信号——输入 (交易日, 500, 30) 的累计残差，输出每只股票残差组合的权重 ω^port")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(LOG) + "\n")

if n_pass != len(checks):
    raise SystemExit(f"有 {len(checks) - n_pass} 项检查未通过，见 {LOG_PATH}")
